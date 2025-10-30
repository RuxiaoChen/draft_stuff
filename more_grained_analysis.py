import pandas as pd
import networkx as nx
# from igraph import Graph
from sklearn.cluster import DBSCAN
from math import radians
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geohash
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from libpysal.weights import Queen  # pip install libpysal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from shapely.geometry import Point
import geopy.distance
# import community as community_louvain   # pip install python-louvain
from collections import Counter
# import contextily as ctx
from collections import Counter
from matplotlib.colors import to_rgba
import random
import folium

def community_detection_geo_louvain(network_dict, distance_scale=500):
    """
    Detect communities using Geo-Louvain (social + geographic hybrid modularity).

    Parameters
    ----------
    network_dict : dict
        Keys = timestamps; values = DataFrames with columns ['group_1','group_2','Latitude','Longitude','type']
        Each node is an 8-character geohash string.
        Latitude/Longitude columns can belong to either node if network_dict already encodes coordinates.
    distance_scale : float (default=500)
        Characteristic distance (meters) controlling spatial decay in edge weight.

    Returns
    -------
    communities : list of lists
        Each sublist is a list of node IDs belonging to one Geo-Louvain community.
    node_to_comm : dict
        Mapping {node_name: community_id}.
    """

    # ---------- Step 1: merge all monthly edges ----------
    all_edges_df = pd.concat(network_dict.values(), ignore_index=True)
    all_edges_df = all_edges_df.drop_duplicates(subset=["group_1", "group_2"])

    # ---------- Step 2: build undirected base graph ----------
    G = nx.Graph()
    for _, row in all_edges_df.iterrows():
        n1, n2 = row["group_1"], row["group_2"]
        G.add_edge(n1, n2)

    nodes = list(G.nodes())
    print(f"Total nodes: {len(nodes)}")

    # ---------- Step 3: decode geohashes ----------
    coords = {}
    for n in nodes:
        try:
            lat, lon = geohash.decode(n)
            coords[n] = (lat, lon)
        except Exception:
            continue
    print(f"Decoded {len(coords)} nodes with valid coordinates.")

    # ---------- Step 4: assign geographic weights ----------
    for u, v in G.edges():
        if u not in coords or v not in coords:
            G[u][v]["weight"] = 1.0
            continue
        d = geopy.distance.distance(coords[u], coords[v]).m  # distance in meters
        # distance-based decay: closer nodes → higher weight
        G[u][v]["weight"] = 1.0 / (1.0 + d / distance_scale)

    # ---------- Step 5: run Louvain algorithm ----------
    partition = community_louvain.best_partition(G, weight="weight")

    # ---------- Step 6: build outputs ----------
    node_to_comm = partition
    communities = []
    for cid in sorted(set(partition.values())):
        comm_nodes = [n for n, c in partition.items() if c == cid]
        communities.append(comm_nodes)

    # ---------- Step 7: summary ----------
    print(f"Detected {len(communities)} Geo-Louvain communities (scale={distance_scale} m).")
    nx.set_node_attributes(G, node_to_comm, name="community")

    return communities, node_to_comm

def community_detection_geographic_census_merge(network_dict, lee_blocks_path,
                                          min_nodes=50, max_blocks=10,randomseed=1):
    """
    Census-block communities with spatial adjacency merge:
    if a block has < min_nodes, iteratively add adjacent blocks
    until total nodes >= min_nodes or total merged blocks >= max_blocks.
    Returns (communities, node_to_comm) with the same types as before.
    """

    # 1) Merge monthly edges and build node list
    all_edges_df = pd.concat(network_dict.values(), ignore_index=True)
    all_edges_df = all_edges_df.drop_duplicates(subset=["group_1", "group_2"])
    G_nx = nx.from_pandas_edgelist(all_edges_df, source="group_1", target="group_2")
    nodes = list(G_nx.nodes())

    # 2) Decode geohash -> point GeoDataFrame
    node_df = pd.DataFrame({"node": nodes})
    node_df["lat"], node_df["lon"] = zip(*node_df["node"].apply(geohash.decode))
    gdf_nodes = gpd.GeoDataFrame(
        node_df, geometry=gpd.points_from_xy(node_df.lon, node_df.lat), crs="EPSG:4326"
    )

    # 3) Load blocks and spatially join nodes -> blocks
    blocks = gpd.read_file(lee_blocks_path).to_crs("EPSG:4326")[["GEOID10", "geometry"]]
    sj = gpd.sjoin(gdf_nodes, blocks, how="left", predicate="within")
    counts = sj.groupby("GEOID10").size().rename("node_count").reset_index()
    g = blocks.merge(counts, on="GEOID10", how="left").fillna({"node_count": 0})

    # Keep only blocks that actually contain nodes 
    g = g[g["node_count"] > 0].reset_index(drop=True)
    if g.empty:
        return [], {}

    # 4) Build Queen contiguity once
    W = Queen.from_dataframe(g)
    neigh = W.neighbors  # dict: idx -> list of neighbor indices

    # 5) Greedy adjacency merge with stopping rules
    assigned = [-1] * len(g)   # community label per block idx
    cid = 0

    for i in range(len(g)):
        if assigned[i] != -1:
            continue
        # start a new group from block i
        group = {i}
        tot_nodes = int(g.loc[i, "node_count"])
        frontier = set(neigh.get(i, []))
        while tot_nodes < min_nodes and len(group) < max_blocks and frontier:
            # choose neighbor with largest node_count that is unassigned and not already in group
            candidates = [j for j in frontier if j < len(g) and assigned[j] == -1 and j not in group]
            if not candidates:
                break
            j_best = max(candidates, key=lambda j: g.loc[j, "node_count"])
            group.add(j_best)
            tot_nodes += int(g.loc[j_best, "node_count"])
            # expand frontier
            frontier.update(neigh.get(j_best, []))
            frontier -= group

        # assign community id to blocks in group
        for j in group:
            assigned[j] = cid
        cid += 1

    g["community_id"] = assigned

    # 6) Dissolve geometries by community_id (purely for reassignment)
    merged_polys = g.dissolve(by="community_id", as_index=False)[["community_id", "geometry"]]

    # 7) Reassign each node to merged polygons and build outputs
    sj2 = gpd.sjoin(gdf_nodes, merged_polys, how="left", predicate="within")
    sj2["community_id"] = sj2["community_id"].astype(str)

    node_to_comm = sj2.set_index("node")["community_id"].dropna().to_dict()
    communities = [list(gr["node"]) for _, gr in sj2.dropna(subset=["community_id"]).groupby("community_id")]

    # 8) Attach attribute back to the graph (same as之前)
    nx.set_node_attributes(G_nx, node_to_comm, name="community")

    return communities, node_to_comm

def community_detection_geographic_census(network_dict, lee_blocks_path):
    """
    Detect communities based on official census block boundaries (Lee County).

    Parameters
    ----------
    network_dict : dict
        Keys = timestamps; values = DataFrames with columns ['group_1', 'group_2', 'type'].
        Each node ID should be an 8-character geohash string.
    lee_blocks_path : str
        Path to the Lee County census block shapefile (.shp).

    Returns
    -------
    communities : list of lists
        Each sublist contains all node IDs (geohash8) belonging to one census block.
    node_to_comm : dict
        Mapping {node_name: community_id (census GEOID10)}.
    """

    # ---------- Step 1: Merge all monthly edges ----------
    all_edges_df = pd.concat(network_dict.values(), ignore_index=True)
    all_edges_df = all_edges_df.drop_duplicates(subset=["group_1", "group_2"])

    # ---------- Step 2: Create undirected graph ----------
    G_nx = nx.from_pandas_edgelist(all_edges_df, source="group_1", target="group_2")
    nodes = list(G_nx.nodes())
    print(f"Total nodes in merged network: {len(nodes)}")

    # ---------- Step 3: Decode geohash to coordinates ----------
    node_df = pd.DataFrame({"node": nodes})
    node_df["lat"], node_df["lon"] = zip(*node_df["node"].apply(geohash.decode))
    gdf_nodes = gpd.GeoDataFrame(
        node_df,
        geometry=gpd.points_from_xy(node_df.lon, node_df.lat),
        crs="EPSG:4326"
    )

    # ---------- Step 4: Load Lee County census blocks ----------
    lee_blocks = gpd.read_file(lee_blocks_path).to_crs("EPSG:4326")

    # ---------- Step 5: Spatial join (assign each node to the block polygon it falls in) ----------
    joined = gpd.sjoin(
        gdf_nodes,
        lee_blocks[["GEOID10", "geometry"]],
        how="left",
        predicate="within"
    )

    # ---------- Step 6: Build node → community mapping ----------
    node_to_comm = joined.set_index("node")["GEOID10"].dropna().to_dict()

    # ---------- Step 7: Group by census block to form community list ----------
    communities = [
        list(gr["node"])
        for _, gr in joined.dropna(subset=["GEOID10"]).groupby("GEOID10")
    ]

    # ---------- Step 8: Summary ----------
    print(f"Detected {len(communities)} census-block communities in Lee County.")
    print(f"Unmatched (outside blocks): {joined['GEOID10'].isna().sum()} nodes")

    # ---------- Step 9: Attach community attribute back to graph ----------
    nx.set_node_attributes(G_nx, node_to_comm, name="community")

    return communities, node_to_comm

def community_detection_geographic(network_dict, eps_meters=500):
    """
    Detect communities based on geographic proximity (distance clustering).

    Parameters
    ----------
    network_dict : dict
        Keys = timestamps; values = DataFrames with columns ['group_1', 'group_2', 'type'].
        Each node ID should be an 8-character geohash string.
    eps_meters : float, optional (default=500)
        Distance threshold for geographic clustering (in meters).

    Returns
    -------
    communities : list of lists
        Each sublist is a list of node IDs belonging to one geographic community.
    node_to_comm : dict
        Mapping {node_name: community_id}.
    """

    # ---------- Step 1: Merge all monthly edges ----------
    all_edges_df = pd.concat(network_dict.values(), ignore_index=True)
    all_edges_df = all_edges_df.drop_duplicates(subset=["group_1", "group_2"])

    # ---------- Step 2: Build undirected graph ----------
    G_nx = nx.from_pandas_edgelist(all_edges_df, source="group_1", target="group_2")
    nodes = list(G_nx.nodes())
    print(f"Total nodes: {len(nodes)}")

    # ---------- Step 3: Decode geohashes to coordinates ----------
    coords = []
    valid_nodes = []
    for n in nodes:
        try:
            lat, lon = geohash.decode(n)
            coords.append((radians(lat), radians(lon)))  # radians for haversine distance
            valid_nodes.append(n)
        except Exception:
            continue
    coords = np.array(coords)
    print(f"Decoded {len(valid_nodes)} valid geohash nodes.")

    # ---------- Step 4: DBSCAN clustering (haversine metric) ----------
    eps_rad = eps_meters / 6371000.0  # convert meters to radians
    db = DBSCAN(eps=eps_rad, min_samples=3, metric='haversine')
    labels = db.fit_predict(coords)

    # ---------- Step 5: Build community mapping ----------
    node_to_comm = {n: int(l) for n, l in zip(valid_nodes, labels) if l != -1}
    communities = []
    for cid in sorted(set(labels)):
        if cid == -1:  # noise points ignored
            continue
        comm_nodes = [n for n, l in node_to_comm.items() if l == cid]
        communities.append(comm_nodes)

    # ---------- Step 6: Summary ----------
    print(f"Detected {len(communities)} geographic communities (eps={eps_meters} m).")

    # ---------- Step 7: Add attributes back to graph ----------
    nx.set_node_attributes(G_nx, node_to_comm, name="community")

    return communities, node_to_comm

def community_detection(network_dict, method="walktrap"):
    """
    Merge monthly social networks and detect communities using the specified algorithm.
    
    Parameters
    ----------
    network_dict : dict
        Keys = timestamps; values = DataFrames with columns ['group_1', 'group_2', 'type'].
    method : str
        One of ['walktrap', 'label_propagation', 'louvain', 'infomap', 'spinglass'].
    
    Returns
    -------
    communities : igraph.clustering.VertexClustering
        Detected community structure.
    node_to_comm : dict
        Mapping {node_name: community_id}.
    """

    # ---------- Step 1: Merge all monthly edges ----------
    all_edges_df = pd.concat(network_dict.values(), ignore_index=True)
    all_edges_df = all_edges_df.drop_duplicates(subset=["group_1", "group_2"])

    # ---------- Step 2: Create undirected NetworkX graph ----------
    G_nx = nx.from_pandas_edgelist(all_edges_df, source="group_1", target="group_2")

    # ---------- Step 3: Convert to igraph ----------
    G_ig = Graph.TupleList(G_nx.edges(), directed=False)

    # ---------- Step 4: Select community detection algorithm ----------
    method = method.lower()
    if method == "walktrap":
        communities = G_ig.community_walktrap().as_clustering()
    elif method == "label_propagation":
        communities = G_ig.community_label_propagation()
    elif method == "louvain":
        communities = G_ig.community_multilevel()
    elif method == "infomap":
        communities = G_ig.community_infomap()
    elif method == "spinglass":
        # Note: Spinglass requires a connected graph; take largest component if needed
        if not G_ig.is_connected():
            G_ig = G_ig.clusters().giant()
        communities = G_ig.community_spinglass()
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose from: "
                         "walktrap, label_propagation, louvain, infomap, spinglass")

    # ---------- Step 5: Map node → community ID ----------
    node_to_comm = {G_ig.vs[i]["name"]: comm_id
                    for comm_id, members in enumerate(communities)
                    for i in members}

    # ---------- Step 6: Add community labels to NetworkX graph ----------
    nx.set_node_attributes(G_nx, node_to_comm, name="community")

    # ---------- Step 7: Print summary ----------
    print(f"Method: {method}")
    print(f"Total nodes: {G_ig.vcount()}")
    print(f"Total edges: {G_ig.ecount()}")
    print(f"Detected communities: {len(communities)}")

    return communities, node_to_comm


def community_analysis(damage_df, node_to_comm, node_num=50):
    """
    Analyze damage composition and building value for communities 
    with more than `node_num` nodes only.
    """

    # ---------- Step 1: Convert node_to_comm dictionary into DataFrame ----------
    comm_map_df = pd.DataFrame(list(node_to_comm.items()), columns=["geohash8", "community_id"])
    comm_map_df["community_id"] = comm_map_df["community_id"].astype(str)

    # ---------- Step 2: Merge community mapping with damage information ----------
    merged = pd.merge(
        comm_map_df,
        damage_df[["damage_geohash", "DamageLevel", "BldgValue", "EstLoss"]],
        how="left",
        left_on="geohash8",
        right_on="damage_geohash"
    ).drop_duplicates(subset="geohash8")

    # Fill missing damage records as "NoDamage"
    merged["DamageLevel"] = merged["DamageLevel"].fillna("NoDamage")

    # ---------- Step 3: Count community sizes ----------
    community_sizes = merged["community_id"].value_counts()

    # Keep only communities with node count > node_num
    large_communities = community_sizes[community_sizes > node_num].index.astype(str)
    merged = merged[merged["community_id"].isin(large_communities)].copy()

    # Sanity check
    if merged.empty:
        print(f"No communities with more than {node_num} nodes found.")
        return pd.DataFrame()

    # ---------- Step 4: Compute proportions of each DamageLevel per community ----------
    count_table = (
        merged.groupby(["community_id", "DamageLevel"])
        .size()
        .unstack(fill_value=0)
    )

    # Normalize by total nodes per community
    proportion_table = count_table.div(count_table.sum(axis=1), axis=0)

    # ---------- Step 5: Compute mean values ----------
    mean_bldg_value = merged.groupby("community_id")["BldgValue"].mean()
    mean_est_loss = merged.groupby("community_id")["EstLoss"].mean()

    # ---------- Step 6: Combine results ----------
    result = proportion_table.copy()
    result["Mean_BldgValue"] = mean_bldg_value
    result["Mean_EstLoss"] = mean_est_loss
    result["Node_Count"] = community_sizes.loc[result.index].astype(int)

    # Sort by node count for easier inspection
    result = result.sort_values("Node_Count", ascending=False)

    return result


def community_decision_analysis(network_dict, node_to_comm, decision_df, node_num=50):
    """
    Compute community-level network and decision metrics,
    considering only communities with more than 50 nodes.
    """

    # ---------- Step 1: Merge all monthly edges ----------
    all_edges_df = pd.concat(network_dict.values(), ignore_index=True)
    all_edges_df = all_edges_df.drop_duplicates(subset=["group_1", "group_2"])

    # Create undirected graph
    G = nx.from_pandas_edgelist(all_edges_df, source="group_1", target="group_2")

    # ---------- Step 2: Attach community and decision info ----------
    nx.set_node_attributes(G, node_to_comm, "community")

    decision_nodes = set(decision_df["geohash8"].unique())
    nx.set_node_attributes(G, {n: (n in decision_nodes) for n in G.nodes}, "has_decision")

    # ---------- Step 3: Filter to communities with >50 nodes ----------
    # Count how many nodes belong to each community
    community_sizes = pd.Series(node_to_comm).value_counts()
    large_communities = set(community_sizes[community_sizes > node_num].index)

    # ---------- Step 4: Compute metrics for each large community ----------
    records = []
    for comm_id in sorted(large_communities):
        nodes_in_comm = [n for n, cid in node_to_comm.items() if cid == comm_id and n in G.nodes]
        subG = G.subgraph(nodes_in_comm)
        num_nodes = subG.number_of_nodes()
        if num_nodes <= node_num:
            continue  # extra safety check

        num_edges = subG.number_of_edges()
        degrees = dict(subG.degree())
        avg_degree = np.mean(list(degrees.values())) if degrees else 0
        density = nx.density(subG)
        avg_clustering = nx.average_clustering(subG) if num_nodes > 1 else 0

        # Ratio of nodes that made a decision
        decisions_in_comm = [n for n in subG.nodes if subG.nodes[n].get("has_decision", False)]
        decision_ratio = len(decisions_in_comm) / num_nodes if num_nodes > 0 else 0

        # ---------- Neighbor adoption ratio ----------
        adoption_ratios = []
        for node in subG.nodes:
            neighbors = list(subG.neighbors(node))
            if not neighbors:
                continue
            adopted_neighbors = sum(subG.nodes[n].get("has_decision", False) for n in neighbors)
            adoption_ratios.append(adopted_neighbors / len(neighbors))
        neighbor_adoption_ratio = np.mean(adoption_ratios) if adoption_ratios else 0

        records.append({
            "community_id": comm_id,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "density": density,
            "avg_clustering": avg_clustering,
            "decision_ratio": decision_ratio,
            "neighbor_adoption_ratio": neighbor_adoption_ratio
        })

    # ---------- Step 5: Combine results ----------
    result_df = pd.DataFrame(records).sort_values("num_nodes", ascending=False).reset_index(drop=True)
    return result_df


def analyze_social_correlation_by_community(network_dict, node_to_comm, decision_df,
                                            min_nodes=1, plot=True):
    """
    For each community: build the induced subgraph, compute per-node neighbor
    decision ratio (using neighbors inside the same community), and compute
    Spearman correlation between has_decision and neighbor_decision_ratio.

    Returns a DataFrame with one row per community.
    """

    # 1) merge monthly edges and build undirected graph
    edges = pd.concat(network_dict.values(), ignore_index=True).drop_duplicates(["group_1","group_2"])
    G = nx.from_pandas_edgelist(edges, "group_1", "group_2")

    # 2) node attributes
    decided = set(decision_df["geohash8"].unique())
    nx.set_node_attributes(G, {n: (n in decided) for n in G.nodes}, "has_decision")

    # 3) group nodes by community
    comm_to_nodes = {}
    for n, cid in node_to_comm.items():
        if n in G:  # keep nodes present in the merged graph
            comm_to_nodes.setdefault(cid, []).append(n)

    # 4) compute per-community correlation
    rows = []
    for cid, nodes in comm_to_nodes.items():
        if len(nodes) < min_nodes:
            continue
        H = G.subgraph(nodes)  # induced subgraph = only intra-community edges

        # per-node neighbor adoption ratio (within the community)
        nbr_ratio, y = [], []
        for u in H.nodes:
            nbrs = list(H.neighbors(u))
            if len(nbrs) == 0:
                continue
            r = np.mean([H.nodes[v]["has_decision"] for v in nbrs])
            nbr_ratio.append(r)
            y.append(int(H.nodes[u]["has_decision"]))

        if len(nbr_ratio) < 3:
            continue

        rho, p = spearmanr(y, nbr_ratio)
        rows.append({
            "community_id": cid,
            "num_nodes": H.number_of_nodes(),
            "num_edges": H.number_of_edges(),
            "spearman_rho": float(rho),
            "p_value": float(p),
            "mean_nbr_ratio": float(np.mean(nbr_ratio)),
            "decision_ratio": float(np.mean(y))
        })

    out = pd.DataFrame(rows).sort_values(["p_value","spearman_rho"], ascending=[True,False])

    if plot and not out.empty:
        plt.figure(figsize=(8,4))
        plt.bar(out["community_id"].astype(str), out["spearman_rho"])
        plt.axhline(0, color="k", lw=0.8)
        plt.ylabel("Spearman rho")
        plt.xlabel("community_id")
        plt.title(f"Neighbor adoption vs decision (per community, min_nodes>{min_nodes})")
        plt.tight_layout()
        plt.show()

    return out



def monthly_rho_lines_id(network_dict, node_to_comm, decision_df,
                      min_valid=10, min_nodes=30, plot=True, plot_cids=None):
    """
    Compute monthly Spearman rho only for communities with >min_nodes.
    
    Optionally, specifies a list of community IDs to plot.
    """
    months = sorted(network_dict.keys())
    plot_cids=[str(cid) for cid in plot_cids]
    # ---- global graph ----
    # Ensure all required libraries (like pandas, networkx, numpy, matplotlib) are imported
    # before running this code in a real environment.
    edges = pd.concat(network_dict.values(), ignore_index=True).drop_duplicates(["group_1","group_2"])
    G = nx.from_pandas_edgelist(edges, "group_1", "group_2")

    # ---- community membership ----
    comm_nodes = {}
    for n, cid in node_to_comm.items():
        if n in G:
            comm_nodes.setdefault(cid, []).append(n)

    # ---- keep only large communities for analysis ----
    comm_sizes = Counter({cid: len(nodes) for cid, nodes in comm_nodes.items()})
    large_comms = [cid for cid, sz in comm_sizes.items() if sz > min_nodes]
    comm_nodes = {cid: comm_nodes[cid] for cid in large_comms}
    print(f"Analyzing {len(large_comms)} communities with >{min_nodes} nodes.")

    rows = []
    for m in months:
        # Assuming decision_df has columns 'decision_date' and 'geohash8'
        decided = set(decision_df.loc[decision_df["decision_date"] <= m, "geohash8"])
        nx.set_node_attributes(G, {n: (n in decided) for n in G.nodes}, "has_decision")

        for cid, nodes in comm_nodes.items():
            nodes_m = [n for n in nodes if n in G]
            if len(nodes_m) < 3:
                rows.append({"month": m, "community_id": cid, "spearman_rho": np.nan})
                continue

            nbr_ratio, y = [], []
            for u in nodes_m:
                nbrs = [v for v in G.neighbors(u) if v in nodes_m]
                if not nbrs:
                    continue
                # G.nodes[v]["has_decision"] is boolean, np.mean will treat True/False as 1/0
                nbr_ratio.append(np.mean([G.nodes[v]["has_decision"] for v in nbrs]))
                y.append(int(G.nodes[u]["has_decision"]))

            if len(nbr_ratio) < min_valid:
                rho = np.nan
            else:
                # Ensure scipy.stats.spearmanr is imported
                rho, _ = spearmanr(y, nbr_ratio)
            rows.append({"month": m, "community_id": cid, "spearman_rho": rho})

    out = pd.DataFrame(rows)

    if plot and not out.empty:
        plt.figure(figsize=(9,6))
        
        # --- NEW LOGIC: Filter communities for plotting ---
        if plot_cids is not None:
            # Filter the results DataFrame to only include the specified community IDs
            plot_df = out[out["community_id"].isin(plot_cids)].copy()
            if plot_df.empty:
                print(f"Warning: No data to plot for specified CIDs: {plot_cids}")
                plt.close()
                return out
        else:
            # If plot_cids is not specified, plot all communities that were analyzed
            plot_df = out.copy()
            
        # --- Plotting ---
        plotted_cids = set()
        for cid, grp in plot_df.groupby("community_id"):
            plt.plot(grp["month"], grp["spearman_rho"], marker="o", label=f"Comm {cid}")
            plotted_cids.add(cid)
            
        plt.axhline(0, color="gray", lw=0.8)
        plt.ylabel("Spearman rho"); plt.xlabel("Month")
        
        # Adjust title based on whether specific CIDs were plotted
        title_cids = f"Specified CIDs: {', '.join(map(str, sorted(list(plotted_cids))))}" if plot_cids else f"Communities with >{min_nodes} nodes"
        plt.title(f"Monthly Spearman rho ({title_cids})")
        
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout(); plt.show()

    return out

def plot_communities_on_map(node_to_comm, base_map_path=None, min_size=30, figsize=(10,10)):
    """
    Visualize geographic communities on a map using node_to_comm.
    Only plot communities whose node count > min_size.

    Parameters
    ----------
    node_to_comm : dict
        Mapping {geohash8: community_id}.
    base_map_path : str, optional
        Path to shapefile for geographic background.
    min_size : int, optional
        Minimum number of nodes a community must have to be plotted.
    figsize : tuple, optional
        Size of matplotlib figure.

    Returns
    -------
    gdf_nodes : GeoDataFrame
        Node points with assigned community_id (filtered by min_size).
    """

    # ---------- Step 1: prepare node DataFrame ----------
    df = pd.DataFrame(list(node_to_comm.items()), columns=["geohash8", "community_id"])
    df["lat"], df["lon"] = zip(*df["geohash8"].apply(geohash.decode))

    # ---------- Step 2: count community sizes ----------
    comm_sizes = Counter(df["community_id"])
    valid_comms = [c for c, s in comm_sizes.items() if s > min_size]
    df = df[df["community_id"].isin(valid_comms)]

    if df.empty:
        print(f"No communities with more than {min_size} nodes.")
        return None

    # ---------- Step 3: make GeoDataFrame ----------
    gdf_nodes = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    # ---------- Step 4: plot ----------
    fig, ax = plt.subplots(figsize=figsize)

    if base_map_path:
        base_map = gpd.read_file(base_map_path).to_crs("EPSG:4326")
        base_map.plot(ax=ax, color="white", edgecolor="lightgray", linewidth=0.5)

    # Assign unique colors to communities
    comm_ids = sorted(gdf_nodes["community_id"].unique())
    n = len(comm_ids)
    cmap = plt.cm.get_cmap("tab20", n)
    colors = [to_rgba(cmap(i)) for i in range(n)]
    color_map = dict(zip(comm_ids, colors))
    gdf_nodes["color"] = gdf_nodes["community_id"].map(color_map)

    for cid, group in gdf_nodes.groupby("community_id"):
        group.plot(ax=ax, color=color_map[cid], markersize=8, label=f"Comm {cid} (n={len(group)})")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Communities with >{min_size} nodes")
    ax.legend(fontsize=8, loc="lower left", markerscale=1.5, ncol=2)
    plt.tight_layout()
    plt.show()

    return gdf_nodes


def plot_communities_on_html_map(node_to_comm, min_size=30, map_file="communities_map.html"):
    """
    Plot communities on an interactive HTML map (Leaflet via folium).
    """
    df = pd.DataFrame(list(node_to_comm.items()), columns=["geohash8", "community_id"])
    df["lat"], df["lon"] = zip(*df["geohash8"].apply(geohash.decode))

    comm_sizes = Counter(df["community_id"])
    valid_comms = [c for c, s in comm_sizes.items() if s > min_size]
    df = df[df["community_id"].isin(valid_comms)]

    if df.empty:
        print(f"No communities with more than {min_size} nodes.")
        return None

    center = [df["lat"].mean(), df["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

    comm_ids = sorted(df["community_id"].unique())
    color_map = {
        cid: f"#{random.randint(0, 0xFFFFFF):06x}" for cid in comm_ids
    }

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color=color_map[row["community_id"]],
            fill=True,
            fill_opacity=0.7,
            popup=f"Community {row['community_id']}",
        ).add_to(fmap)

    legend_html = "<div style='position: fixed; bottom: 30px; left: 30px; width: 200px; \
                    background-color: white; padding: 10px; border:1px solid gray;'> \
                    <b>Communities</b><br>"
    for cid, col in color_map.items():
        legend_html += f"<span style='background-color:{col};width:12px;height:12px;display:inline-block;'></span> Comm {cid}<br>"
    legend_html += "</div>"
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(map_file)
    print(f"Map saved to: {map_file}")
    return fmap


def animate_decision_network(decision_df, households_df, network_dict, save_path="network_evolution.gif"):
    """
    Visualize monthly household networks over geographic space as an animation.

    Parameters
    ----------
    decision_df : pd.DataFrame
        ['geohash8', 'decision_date', 'decision_type']
    households_df : pd.DataFrame
        Must contain ['geohash8', 'lat', 'lon']
    network_dict : dict[datetime -> pd.DataFrame]
        Each value must have ['group_1', 'group_2', 'type']
    save_path : str
        Output path for the animation GIF.
    """

    # ------------------------------------------------------
    # 1. Prepare coordinate lookup and all households set
    # ------------------------------------------------------
    coord_map = {r['geohash8']: (r['lon'], r['lat']) for _, r in households_df.iterrows()}
    all_households = set(households_df['geohash8'].unique())  # 固定的家庭集合

    # ------------------------------------------------------
    # 2. Prepare month list
    # ------------------------------------------------------
    months = sorted(list(network_dict.keys()))

    # ------------------------------------------------------
    # 3. Setup figure
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Household Decision Network Evolution")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # color map for decision
    color_map = {"repair": "red", "sales": "green", "none": "gray"}
    edge_color_map = {0: "black", 1: "blue"}

    # ------------------------------------------------------
    # 4. Draw each frame
    # ------------------------------------------------------
    def update(frame_idx):
        ax.clear()
        month = months[frame_idx]
        net_df = network_dict[month]

        ax.set_title(f"Network - {pd.to_datetime(month).strftime('%Y-%m')}")

        # build graph
        G = nx.Graph()
        for _, r in net_df.iterrows():
            G.add_edge(r['group_1'], r['group_2'], type=r['type'])

        # determine decisions in this month
        month_start = pd.Timestamp(month)
        month_end = month_start + pd.DateOffset(months=1)
        month_decisions = decision_df[
            (decision_df['decision_date'] >= month_start) &
            (decision_df['decision_date'] < month_end)
        ]

        # classify node colors for ALL households
        node_colors = {}
        for household in all_households:  
            if household not in coord_map:
                continue
            decs = month_decisions.loc[month_decisions['geohash8'] == household, 'decision_type']
            if len(decs) == 0:
                node_colors[household] = color_map["none"]
            else:
                node_colors[household] = color_map[decs.iloc[0]]

        # draw edges by type
        for etype, color in edge_color_map.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == etype]
            if edges:
                edge_xy = [
                    ([coord_map[u][0], coord_map[v][0]],
                     [coord_map[u][1], coord_map[v][1]])
                    for u, v in edges if u in coord_map and v in coord_map
                ]
                for x, y in edge_xy:
                    ax.plot(x, y, color=color, alpha=0.4, linewidth=1)

        # draw ALL nodes (not just those in the network)
        xs, ys, cs = [], [], []
        for household, color in node_colors.items():
            if household in coord_map:
                xs.append(coord_map[household][0])
                ys.append(coord_map[household][1])
                cs.append(color)
        ax.scatter(xs, ys, c=cs, s=10, edgecolors="k", linewidths=0.3)

        ax.set_xlim(min(households_df["lon"]) - 0.01, max(households_df["lon"]) + 0.01)
        ax.set_ylim(min(households_df["lat"]) - 0.01, max(households_df["lat"]) + 0.01)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # ------------------------------------------------------
    # 5. Create animation
    # ------------------------------------------------------
    ani = animation.FuncAnimation(fig, update, frames=len(months), interval=1200, repeat=False)

    ani.save(save_path, writer="pillow", fps=1)
    plt.close(fig)
    print(f"✅ Saved animation to {save_path}")