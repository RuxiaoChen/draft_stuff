import pandas as pd
import geohash
import pdb
import numpy as np
import glob
from dateutil.relativedelta import relativedelta
import os
import networkx as nx


def load_decision_data(bbox=(-82.0, 26.48, -81.92, 26.52)):
    """
    Returns:
      decision_df : rows with a VALID decision timestamp (repair or sales)
      households_df: unique households (geohash8) within bbox, even if they NEVER made a decision

    Notes:
    - sales['first_sale_in_period'] may contain 0/'' -> treated as "no decision yet" but we still keep the household.
    - decision_df has columns: [geohash8, decision_date, decision_type]
    - households_df has columns: [geohash8, lat, lon, has_repair_decision, has_sales_decision]
    """
    minx, miny, maxx, maxy = bbox

    # -----------------------------
    # 1) Load raw CSVs
    # -----------------------------
    repair = pd.read_csv(
        'decision_data/repair_coords_mapped_to_sales.csv',
        parse_dates=['record_date']
    )
    sales = pd.read_csv(
        'decision_data/sales_data_all.csv',
        dayfirst=False
    )

    # -----------------------------
    # 2) BBox filter
    # -----------------------------
    repair = repair[
        repair['original_lon'].between(minx, maxx) &
        repair['original_lat'].between(miny, maxy)
    ].copy()
    sales = sales[
        sales['lon'].between(minx, maxx) &
        sales['lat'].between(miny, maxy)
    ].copy()

    # -----------------------------
    # 3) Geohash8 encoding
    # -----------------------------
    repair['geohash8'] = repair.apply(
        lambda r: geohash.encode(r['original_lat'], r['original_lon'], precision=8), axis=1
    )
    sales['geohash8'] = sales.apply(
        lambda r: geohash.encode(r['lat'], r['lon'], precision=8), axis=1
    )

    # -----------------------------
    # 4) Build households_df (unique households inside bbox)
    #    Union of geohashes from repair & sales
    # -----------------------------
    # decode one time so households have lat/lon center (optional but handy)
    def _decode(gh):
        try:
            lat, lon = geohash.decode(gh)
            return pd.Series([lat, lon])
        except Exception:
            return pd.Series([np.nan, np.nan])

    all_hashes = pd.Index(repair['geohash8']).union(pd.Index(sales['geohash8']))
    households_df = pd.DataFrame({'geohash8': all_hashes})
    households_df[['lat', 'lon']] = households_df['geohash8'].apply(_decode)

    # -----------------------------
    # 5) Parse decision timestamps properly
    #    - repair: record_date already parsed above
    #    - sales: first_sale_in_period may be 0 / '' / date-like string
    # -----------------------------
    # Normalize sales decision column into a datetime; invalid -> NaT
    sales['decision_date'] = pd.to_datetime(
        sales['first_sale_in_period'],
        errors='coerce', infer_datetime_format=True
    )
    repair['decision_date'] = pd.to_datetime(
        repair['record_date'],
        errors='coerce', infer_datetime_format=True
    )

    # -----------------------------
    # 6) Build decision_df (ONLY rows with a valid decision_date)
    # -----------------------------
    repair_dec = repair[['geohash8', 'decision_date']].dropna(subset=['decision_date']).copy()
    sales_dec  = sales [['geohash8', 'decision_date']].dropna(subset=['decision_date']).copy()

    repair_dec['decision_type'] = 'repair'
    sales_dec ['decision_type']  = 'sales'

    decision_df = pd.concat([repair_dec, sales_dec], ignore_index=True)

    # -----------------------------
    # 7) Add boolean flags to households_df: who ever made a decision?
    # -----------------------------
    households_df = households_df.merge(
        repair_dec[['geohash8']].drop_duplicates().assign(has_repair_decision=True),
        on='geohash8', how='left'
    ).merge(
        sales_dec[['geohash8']].drop_duplicates().assign(has_sales_decision=True),
        on='geohash8', how='left'
    )

    households_df['has_repair_decision'] = households_df['has_repair_decision'].fillna(False)
    households_df['has_sales_decision']  = households_df['has_sales_decision'].fillna(False)

    # -----------------------------
    # 8) Logging
    # -----------------------------
    # print(f"[INFO] households in bbox: {len(households_df)} "
    #       f"(repair nodes: {repair['geohash8'].nunique()}, sales nodes: {sales['geohash8'].nunique()})")
    # print(f"[INFO] decision rows kept (valid timestamps): {len(decision_df)} "
    #       f"(repair: {len(repair_dec)}, sales: {len(sales_dec)})")

    return decision_df, households_df


def load_network_files(folder='social_network', bbox=(-82.0, 26.48, -81.92, 26.52)):
    minx, miny, maxx, maxy = bbox
    files = sorted(glob.glob(os.path.join(folder, 'Group_social_network_*.csv')))
    network_dict = {}

    def _valid_gh8(s: str) -> bool:
        if not isinstance(s, str): return False
        s = s.strip().lower()
        return len(s) == 8 and all(ch in "0123456789bcdefghjkmnpqrstuvwxyz" for ch in s)

    def _safe_decode(s):
        try:
            lat, lon = geohash.decode(s)
            return lat, lon
        except Exception:
            return np.nan, np.nan

    for f in files:

        date_str = os.path.splitext(os.path.basename(f))[0].split('_')[-1]
        net_date = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(net_date):
            print(f"[WARN] skip file (bad date): {f}")
            continue

        df = pd.read_csv(f)

        need_cols = {'group_1', 'group_2', 'type'}
        if not need_cols.issubset(df.columns):
            print(f"[WARN] skip file (missing cols): {f}")
            continue
        df = df[['group_1', 'group_2', 'type']].copy()

        df = df.dropna(subset=['group_1', 'group_2'])
        df['group_1'] = df['group_1'].astype(str).str.strip().str.lower()
        df['group_2'] = df['group_2'].astype(str).str.strip().str.lower()

        df = df[df['group_1'].map(_valid_gh8) & df['group_2'].map(_valid_gh8)]
        if df.empty:
            print(f"[INFO] {f}: 0 edges after GH8 sanitize")
            network_dict[net_date] = df[['group_1','group_2','type']]
            continue

        g1_latlon = df['group_1'].map(_safe_decode)
        g2_latlon = df['group_2'].map(_safe_decode)
        df['g1_lat'] = [x[0] for x in g1_latlon]
        df['g1_lon'] = [x[1] for x in g1_latlon]
        df['g2_lat'] = [x[0] for x in g2_latlon]
        df['g2_lon'] = [x[1] for x in g2_latlon]

        df = df.dropna(subset=['g1_lat','g1_lon','g2_lat','g2_lon'])


        df = df[
            (df['g1_lon'].between(minx, maxx)) &
            (df['g1_lat'].between(miny, maxy)) &
            (df['g2_lon'].between(minx, maxx)) &
            (df['g2_lat'].between(miny, maxy))
        ]


        df = df[['group_1', 'group_2', 'type']].reset_index(drop=True)
        network_dict[net_date] = df
        nodes = set(df['group_1']).union(set(df['group_2']))
        print(f"[INFO] {f}: kept {len(df)} edges, {len(nodes)} nodes in bbox")

    return network_dict


import networkx as nx
import numpy as np
import pandas as pd

def analyze_decision_network_strength(decision_df, households_df, network_dict, month_window=1):
    """
    For each monthly social network, compare network structure between:
      - households that made a decision in a given time window
      - households that did NOT make a decision

    Added outputs:
      - n_nodes, n_edges, avg_clustering
      - removed avg_neighbor_degree
    """

    records = []

    for date, net_df in network_dict.items():
        # ----------------------------
        # Build graph for this month
        # ----------------------------
        G = nx.Graph()
        for _, r in net_df.iterrows():
            G.add_edge(r['group_1'], r['group_2'], weight=1, type=r['type'])

        all_nodes = set(G.nodes())

        # ----------------------------
        # Decision window
        # ----------------------------
        month_start = pd.Timestamp(date)
        month_end = month_start + pd.DateOffset(months=month_window)

        decided_nodes = set(
            decision_df.loc[
                (decision_df['decision_date'] >= month_start) &
                (decision_df['decision_date'] < month_end),
                'geohash8'
            ]
        )

        all_households = set(households_df['geohash8'])
        decided_nodes = decided_nodes & all_households
        non_decided_nodes = all_households - decided_nodes

        if not decided_nodes or not non_decided_nodes:
            continue

        # ----------------------------
        # Compute metrics
        # ----------------------------
        deg = dict(G.degree())
        betw = nx.betweenness_centrality(G, normalized=True)
        clustering = nx.clustering(G)

        def summarize(nodes, label):
            sub_nodes = [n for n in nodes if n in deg]
            if not sub_nodes:
                return None
            subG = G.subgraph(sub_nodes)
            return {
                'date': date,
                'group': label,
                'n_nodes': len(sub_nodes),
                'n_edges': subG.number_of_edges(),
                'avg_degree': np.mean([deg[n] for n in sub_nodes]),
                'avg_betweenness': np.mean([betw[n] for n in sub_nodes]),
                'avg_clustering': np.mean([clustering[n] for n in sub_nodes]),
                'density': nx.density(subG) if len(sub_nodes) > 1 else 0
            }

        rec_d = summarize(decided_nodes, 'decided')
        rec_n = summarize(non_decided_nodes, 'non_decided')

        if rec_d: records.append(rec_d)
        if rec_n: records.append(rec_n)

        # print monthly summary
        print(f"{date.date()} | decided: {rec_d['n_nodes']} nodes, {rec_d['n_edges']} edges "
              f"| non-decided: {rec_n['n_nodes']} nodes, {rec_n['n_edges']} edges")

    # ----------------------------
    # Combine results
    # ----------------------------
    result = pd.DataFrame([r for r in records if r])
    print(
        result.groupby('group')[['avg_degree', 'avg_betweenness',
                                 'avg_clustering', 'density']].mean()
    )
    return result



def analyze_overall_decision_network(decision_df, households_df, network_dict):
    """
    Combine all monthly networks into a single large network
    and compare structural metrics between decided and non-decided households.

    Outputs:
      - avg_degree
      - avg_betweenness
      - avg_clustering
      - density
      - n_nodes, n_edges (unique & weighted)
    """

    # -------------------------------------------------
    # 1. Combine all monthly edges into one DataFrame
    # -------------------------------------------------
    all_edges = []
    for date, df in network_dict.items():
        if {'group_1','group_2'}.issubset(df.columns):
            tmp = df[['group_1','group_2','type']].dropna(subset=['group_1','group_2']).copy()
            def safe_sort(r):
                g1, g2 = str(r['group_1']).strip(), str(r['group_2']).strip()
                if not g1 or not g2 or g1 == 'nan' or g2 == 'nan':
                    return pd.Series([None, None])
                return pd.Series(sorted([g1, g2]))

            tmp[['u','v']] = tmp.apply(safe_sort, axis=1)
            tmp = tmp.dropna(subset=['u','v'])
            all_edges.append(tmp)
    if not all_edges:
        raise ValueError("No valid network data found.")
    all_edges = pd.concat(all_edges, ignore_index=True)

    # aggregate duplicates into weighted edges
    agg = (all_edges.groupby(['u','v','type'])
                    .size().reset_index(name='weight'))

    print(f"[INFO] Combined network edges: {len(all_edges)} rows, {len(agg)} unique pairs")

    # -------------------------------------------------
    # 2. Build full graph
    # -------------------------------------------------
    G = nx.Graph()
    for _, r in agg.iterrows():
        G.add_edge(r['u'], r['v'], type=r['type'], weight=int(r['weight']))

    print(f"[INFO] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} unique edges")

    # -------------------------------------------------
    # 3. Decide who made decisions (across all months)
    # -------------------------------------------------
    decided_nodes = set(decision_df['geohash8'].dropna())
    all_households = set(households_df['geohash8'].dropna())
    decided_nodes = decided_nodes & all_households
    non_decided_nodes = all_households - decided_nodes

    print(f"[INFO] Decided households: {len(decided_nodes)}, Non-decided: {len(non_decided_nodes)}")

    # -------------------------------------------------
    # 4. Compute network metrics
    # -------------------------------------------------
    deg = dict(G.degree())
    betw = nx.betweenness_centrality(G, normalized=True)
    clustering = nx.clustering(G)

    def summarize(nodes, label):
        sub_nodes = [n for n in nodes if n in G]
        if not sub_nodes:
            return None
        subG = G.subgraph(sub_nodes)
        return {
            'group': label,
            'n_nodes': len(sub_nodes),
            'n_edges_unique': subG.number_of_edges(),
            'n_edges_weighted': sum(d.get('weight',1) for _,_,d in subG.edges(data=True)),
            'avg_degree': np.mean([deg[n] for n in sub_nodes]),
            'avg_betweenness': np.mean([betw[n] for n in sub_nodes]),
            'avg_clustering': np.mean([clustering[n] for n in sub_nodes]),
            'density': nx.density(subG) if len(sub_nodes) > 1 else 0.0
        }

    rec_d = summarize(decided_nodes, 'decided')
    rec_n = summarize(non_decided_nodes, 'non_decided')

    # -------------------------------------------------
    # 5. Combine and output
    # -------------------------------------------------
    result = pd.DataFrame([r for r in [rec_d, rec_n] if r])
    print("\n=== Overall Network Statistics ===")
    print(result.set_index('group')[['n_nodes','n_edges_unique','avg_degree',
                                    'avg_betweenness','avg_clustering','density']])
    return result


def analyze_overall_decision_network2(decision_df, households_df, network_dict):
    import networkx as nx
    import numpy as np
    import pandas as pd
    from networkx.algorithms.community.quality import modularity

    all_edges = []

    for date, df in network_dict.items():
        if {'group_1','group_2'}.issubset(df.columns):
            tmp = df[['group_1','group_2','type']].dropna(subset=['group_1','group_2']).copy()

            u_list, v_list = [], []
            for g1, g2 in zip(tmp['group_1'], tmp['group_2']):
                g1, g2 = str(g1).strip(), str(g2).strip()
                if not g1 or not g2 or g1 in ['nan', 'None'] or g2 in ['nan', 'None']:
                    continue
                u, v = sorted([g1, g2])
                u_list.append(u)
                v_list.append(v)

            tmp = tmp.iloc[:len(u_list)].copy()
            tmp['u'] = u_list
            tmp['v'] = v_list
            all_edges.append(tmp)

    if not all_edges:
        raise ValueError("No valid network edges found.")
    all_edges = pd.concat(all_edges, ignore_index=True)

    # ---- aggregate duplicates into weighted edges ----
    agg = (
        all_edges.groupby(['u','v','type'])
        .size()
        .reset_index(name='weight')
    )

    print(f"[INFO] Combined edges: {len(all_edges)}, unique pairs: {len(agg)}")

    # ---- build full graph ----
    G = nx.Graph()
    for _, r in agg.iterrows():
        G.add_edge(r['u'], r['v'], type=r['type'], weight=int(r['weight']))
    print(f"[INFO] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} unique edges")

    # ---- determine deciders ----
    decided = set(decision_df['geohash8'].dropna())
    all_hh = set(households_df['geohash8'].dropna())
    decided = decided & all_hh
    non_decided = all_hh - decided
    print(f"[INFO] Decided: {len(decided)}, Non-decided: {len(non_decided)}")

    # ---- metrics ----
    deg = dict(G.degree())
    betw = nx.betweenness_centrality(G, normalized=True)
    cluster = nx.clustering(G)

    def summarize(nodes, label):
        valid = [n for n in nodes if n in G]
        if not valid:
            return None
        subG = G.subgraph(valid)
        # modularity needs a partition list of sets
        # partition = [set(valid), set(G.nodes()) - set(valid)]
        # mod = modularity(G, partition, weight='weight')
        return {
            'group': label,
            'n_nodes': len(valid),
            'n_edges_unique': subG.number_of_edges(),
            'n_edges_weighted': sum(d.get('weight',1) for _,_,d in subG.edges(data=True)),
            'avg_degree': np.mean([deg[n] for n in valid]),
            'avg_betweenness': np.mean([betw[n] for n in valid]),
            'avg_clustering': np.mean([cluster[n] for n in valid]),
            'density': nx.density(subG) if len(valid) > 1 else 0.0,
            # 'modularity': mod
        }

    rec_d = summarize(decided, 'decided')
    rec_n = summarize(non_decided, 'non_decided')

    result = pd.DataFrame([r for r in [rec_d, rec_n] if r])
    print(result.set_index('group')[['n_nodes','n_edges_unique','avg_degree',
                                    'avg_betweenness','avg_clustering','density']])
    return result



def summarize_decider_contrast(decision_df, households_df, network_dict):
    import networkx as nx
    import numpy as np
    import pandas as pd

    # ---------- 1. Merge all edges ----------
    all_edges = []
    for _, df in network_dict.items():
        if {'group_1','group_2'}.issubset(df.columns):
            tmp = df[['group_1','group_2']].dropna().copy()
            tmp['u'] = tmp.apply(lambda r: min(str(r['group_1']).strip(), str(r['group_2']).strip()), axis=1)
            tmp['v'] = tmp.apply(lambda r: max(str(r['group_1']).strip(), str(r['group_2']).strip()), axis=1)
            tmp = tmp[(tmp['u']!='') & (tmp['v']!='') & (tmp['u']!='nan') & (tmp['v']!='nan')]
            all_edges.append(tmp[['u','v']])
    if not all_edges:
        raise ValueError("No edges found.")
    all_edges = pd.concat(all_edges, ignore_index=True)
    agg = all_edges.value_counts(['u','v']).reset_index(name='weight')

    G = nx.Graph()
    for _, r in agg.iterrows():
        G.add_edge(r['u'], r['v'], weight=int(r['weight']))

    # ---------- 2. Label decided ----------
    all_hh = set(households_df['geohash8'].dropna().astype(str))
    decided = set(decision_df['geohash8'].dropna().astype(str)) & all_hh
    for n in G.nodes():
        G.nodes[n]['decided'] = 1 if n in decided else 0

    # ---------- 3. Node-level metrics ----------
    deg = dict(G.degree())
    clustering = nx.clustering(G)
    try:
        core = nx.core_number(G)
    except Exception:
        core = {n: 0 for n in G.nodes()}
    pr = nx.pagerank(G, alpha=0.85)

    neighbor_decided_frac = {}
    for n in G.nodes():
        nbrs = list(G.neighbors(n))
        if not nbrs:
            neighbor_decided_frac[n] = 0.0
        else:
            neighbor_decided_frac[n] = np.mean([G.nodes[v]['decided'] for v in nbrs])

    per_node = pd.DataFrame({
        'node': list(G.nodes()),
        'decided': [G.nodes[n]['decided'] for n in G.nodes()],
        'degree': [deg[n] for n in G.nodes()],
        'clustering': [clustering.get(n,0.0) for n in G.nodes()],
        'kcore': [core.get(n,0) for n in G.nodes()],
        'pagerank': [pr.get(n,0.0) for n in G.nodes()],
        'neighbor_decided_frac': [neighbor_decided_frac[n] for n in G.nodes()],
    })

    # ---------- 4. Group summaries ----------
    def agg_block(df, label):
        nodes = list(df['node'])
        subG = G.subgraph(nodes)
        return pd.Series({
            'n_nodes': len(nodes),
            'n_edges': subG.number_of_edges(),
            'degree_mean': df['degree'].mean(),
            'degree_median': df['degree'].median(),
            'clust_mean': df['clustering'].mean(),
            'kcore_mean': df['kcore'].mean(),
            'pagerank_mean': df['pagerank'].mean(),
            'nbr_decided_frac_mean': df['neighbor_decided_frac'].mean(),
        })

    summary = []
    for label, group_df in per_node.groupby(per_node['decided'].map({0:'non_decided',1:'decided'})):
        summary.append(agg_block(group_df, label))
    summary = pd.DataFrame(summary, index=['decided','non_decided'])

    # ---------- 5. Global metrics ----------
    edges = list(G.edges())
    E = sum((G.nodes[u]['decided'] != G.nodes[v]['decided']) for u,v in edges)
    I = len(edges) - E
    cross_share = E / len(edges) if edges else 0.0
    EI_index = (E - I) / (E + I) if (E + I) > 0 else np.nan

    try:
        mixing_matrix = nx.attribute_mixing_matrix(G, 'decided', normalized=True)
    except Exception:
        mixing_matrix = np.nan
    try:
        assort = nx.attribute_assortativity_coefficient(G, 'decided')
    except Exception:
        assort = np.nan

    globals_ = {
        'total_edges': G.number_of_edges(),
        'cross_edge_share': cross_share,
        'EI_index': EI_index,
        'mixing_matrix': mixing_matrix,
        'assortativity_decided': assort
    }

    # ---------- 6. Pretty print ----------
    print("Globals:")
    print(f"  Total edges (whole network): {G.number_of_edges()}")
    print(f"  Cross-edge share: {cross_share:.3f}")
    print(f"  E–I Index: {EI_index:.3f}")
    print(f"\n  Mixing matrix (rows/cols = non_decided[0], decided[1]):\n")
    print(pd.DataFrame(
        mixing_matrix,
        index=["from non-decided", "from decided"],
        columns=["to non-decided", "to decided"]
    ).applymap(lambda x: f"{x:.3f}"))

    print("\nSummary by group:\n", summary)
    return per_node, summary.reset_index().rename(columns={'index':'group'}), globals_


def summarize_decider_contrast_by_month(decision_df, households_df, network_dict):
    """
    For each month in network_dict:
      - Build that month's social network
      - Compute decider vs non-decider contrast metrics
      - Print full summary (globals + group stats)
    """

    import networkx as nx
    import numpy as np
    import pandas as pd

    for date, df in network_dict.items():
        print(f"\n===============================")
        print(f"Month: {date}")
        print(f"===============================")

        # ---------- 1. Build graph ----------
        if not {'group_1', 'group_2'}.issubset(df.columns):
            print(f"[WARN] Missing columns in {date}")
            continue

        tmp = df[['group_1', 'group_2']].dropna().copy()
        tmp['u'] = tmp.apply(lambda r: min(str(r['group_1']).strip(), str(r['group_2']).strip()), axis=1)
        tmp['v'] = tmp.apply(lambda r: max(str(r['group_1']).strip(), str(r['group_2']).strip()), axis=1)
        tmp = tmp[(tmp['u']!='') & (tmp['v']!='') & (tmp['u']!='nan') & (tmp['v']!='nan')]

        if tmp.empty:
            print(f"[INFO] No valid edges for {date}")
            continue

        agg = tmp.value_counts(['u', 'v']).reset_index(name='weight')

        G = nx.Graph()
        for _, r in agg.iterrows():
            G.add_edge(r['u'], r['v'], weight=int(r['weight']))

        # ---------- 2. Label decided ----------
        all_hh = set(households_df['geohash8'].dropna().astype(str))
        decided = set(decision_df['geohash8'].dropna().astype(str)) & all_hh
        for n in G.nodes():
            G.nodes[n]['decided'] = 1 if n in decided else 0

        # ---------- 3. Node-level metrics ----------
        deg = dict(G.degree())
        clustering = nx.clustering(G)
        try:
            core = nx.core_number(G)
        except Exception:
            core = {n: 0 for n in G.nodes()}
        pr = nx.pagerank(G, alpha=0.85)

        neighbor_decided_frac = {}
        for n in G.nodes():
            nbrs = list(G.neighbors(n))
            if not nbrs:
                neighbor_decided_frac[n] = 0.0
            else:
                neighbor_decided_frac[n] = np.mean([G.nodes[v]['decided'] for v in nbrs])

        per_node = pd.DataFrame({
            'node': list(G.nodes()),
            'decided': [G.nodes[n]['decided'] for n in G.nodes()],
            'degree': [deg[n] for n in G.nodes()],
            'clustering': [clustering.get(n,0.0) for n in G.nodes()],
            'kcore': [core.get(n,0) for n in G.nodes()],
            'pagerank': [pr.get(n,0.0) for n in G.nodes()],
            'neighbor_decided_frac': [neighbor_decided_frac[n] for n in G.nodes()],
        })

        # ---------- 4. Group summaries ----------
        def agg_block(df, label):
            nodes = list(df['node'])
            subG = G.subgraph(nodes)
            return pd.Series({
                'n_nodes': len(nodes),
                'n_edges': subG.number_of_edges(),
                'degree_mean': df['degree'].mean(),
                'degree_median': df['degree'].median(),
                'clust_mean': df['clustering'].mean(),
                'kcore_mean': df['kcore'].mean(),
                'pagerank_mean': df['pagerank'].mean(),
                'nbr_decided_frac_mean': df['neighbor_decided_frac'].mean(),
            })

        summary = []
        for label, group_df in per_node.groupby(per_node['decided'].map({0:'non_decided',1:'decided'})):
            summary.append(agg_block(group_df, label))
        summary = pd.DataFrame(summary, index=['decided','non_decided'])

        # ---------- 5. Global metrics ----------
        edges = list(G.edges())
        E = sum((G.nodes[u]['decided'] != G.nodes[v]['decided']) for u,v in edges)
        I = len(edges) - E
        cross_share = E / len(edges) if edges else 0.0
        EI_index = (E - I) / (E + I) if (E + I) > 0 else np.nan

        try:
            mixing_matrix = nx.attribute_mixing_matrix(G, 'decided', normalized=True)
        except Exception:
            mixing_matrix = np.nan
        try:
            assort = nx.attribute_assortativity_coefficient(G, 'decided')
        except Exception:
            assort = np.nan

        # ---------- 6. Print results ----------
        print("Globals:")
        print(f"  Cross-edge share: {cross_share:.3f}")
        print(f"  E–I Index: {EI_index:.3f}")
        print(f"\n  Mixing matrix (rows/cols = non_decided[0], decided[1]):\n")
        print(pd.DataFrame(
            mixing_matrix,
            index=["from non-decided", "from decided"],
            columns=["to non-decided", "to decided"]
        ).applymap(lambda x: f"{x:.3f}"))

        print("\nSummary by group:\n", summary)


def analyze_overall_decision_network2_by_month(decision_df, households_df, network_dict):
    import networkx as nx
    import numpy as np
    import pandas as pd
    from networkx.algorithms.community.quality import modularity

    for date, df in network_dict.items():
        print(f"\n===============================")
        print(f"Month: {date}")
        print(f"===============================")

        # ---- clean edges ----
        if not {'group_1', 'group_2'}.issubset(df.columns):
            print(f"[WARN] Missing columns in {date}")
            continue

        tmp = df[['group_1', 'group_2', 'type']].dropna(subset=['group_1', 'group_2']).copy()
        u_list, v_list = [], []
        for g1, g2 in zip(tmp['group_1'], tmp['group_2']):
            g1, g2 = str(g1).strip(), str(g2).strip()
            if not g1 or not g2 or g1 in ['nan', 'None'] or g2 in ['nan', 'None']:
                continue
            u, v = sorted([g1, g2])
            u_list.append(u)
            v_list.append(v)

        if not u_list:
            print(f"[INFO] No valid edges for {date}")
            continue

        tmp = tmp.iloc[:len(u_list)].copy()
        tmp['u'], tmp['v'] = u_list, v_list

        # ---- aggregate duplicates into weighted edges ----
        agg = tmp.groupby(['u', 'v', 'type']).size().reset_index(name='weight')

        # ---- build monthly graph ----
        G = nx.Graph()
        for _, r in agg.iterrows():
            G.add_edge(r['u'], r['v'], type=r['type'], weight=int(r['weight']))

        print(f"[INFO] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} unique edges")

        # ---- determine deciders ----
        decided = set(decision_df['geohash8'].dropna()) & set(households_df['geohash8'].dropna())
        non_decided = set(households_df['geohash8'].dropna()) - decided
        print(f"[INFO] Decided: {len(decided)}, Non-decided: {len(non_decided)}")

        # ---- metrics ----
        deg = dict(G.degree())
        betw = nx.betweenness_centrality(G, normalized=True)
        cluster = nx.clustering(G)

        def summarize(nodes, label):
            valid = [n for n in nodes if n in G]
            if not valid:
                return None
            subG = G.subgraph(valid)
            partition = [set(valid), set(G.nodes()) - set(valid)]
            try:
                mod = modularity(G, partition, weight='weight')
            except Exception:
                mod = np.nan
            return {
                'group': label,
                'n_nodes': len(valid),
                'n_edges_unique': subG.number_of_edges(),
                'n_edges_weighted': sum(d.get('weight', 1) for _, _, d in subG.edges(data=True)),
                'avg_degree': np.mean([deg[n] for n in valid]),
                'avg_betweenness': np.mean([betw[n] for n in valid]),
                'avg_clustering': np.mean([cluster[n] for n in valid]),
                'density': nx.density(subG) if len(valid) > 1 else 0.0,
            }

        rec_d = summarize(decided, 'decided')
        rec_n = summarize(non_decided, 'non_decided')

        result = pd.DataFrame([r for r in [rec_d, rec_n] if r])
        print(result.set_index('group')[[
            'n_nodes', 'n_edges_unique', 'avg_degree',
            'avg_betweenness', 'avg_clustering', 'density'
        ]])
