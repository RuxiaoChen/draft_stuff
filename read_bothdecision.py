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
    print(f"[INFO] households in bbox: {len(households_df)} "
          f"(repair nodes: {repair['geohash8'].nunique()}, sales nodes: {sales['geohash8'].nunique()})")
    print(f"[INFO] decision rows kept (valid timestamps): {len(decision_df)} "
          f"(repair: {len(repair_dec)}, sales: {len(sales_dec)})")

    return decision_df, households_df


def load_network_files(folder='social_network', bbox=(-82.0, 26.48, -81.92, 26.52)):
    minx, miny, maxx, maxy = bbox
    files = sorted(glob.glob(os.path.join(folder, 'Group_social_network_*.csv')))
    network_dict = {}

    def _valid_gh8(s: str) -> bool:
        # 只接受 8 位 [0-9b-hj-km-np-z]（geohash 不含 a i l o）
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
        print(f"[INFO] {f}: kept {len(df)} edges in bbox {bbox}")

    return network_dict


def analyze_decision_network_strength(decision_df, households_df, network_dict, month_window=1):
    """
    Compare network structural metrics between:
      - households that made a decision in a given time window, and
      - households that did NOT make a decision
    using the household base table as the full population.

    Inputs
    -------
    decision_df : DataFrame
        Contains ['geohash8', 'decision_date', 'decision_type']
    households_df : DataFrame
        All households inside bbox, including those with no decision
    network_dict : dict[datetime -> DataFrame]
        Each value has columns ['group_1', 'group_2', 'type']
    month_window : int
        Number of months after `date` to define decision window

    Returns
    -------
    result_df : DataFrame
        Aggregated network metrics by group ('decided' / 'non_decided')
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
        # Decide who made a decision in this window
        # ----------------------------
        month_start = pd.Timestamp(date)
        month_end = month_start + pd.DateOffset(months=month_window)

        # households that made a decision within [date, date+window)
        decided_nodes = set(
            decision_df.loc[
                (decision_df['decision_date'] >= month_start) &
                (decision_df['decision_date'] < month_end),
                'geohash8'
            ]
        )

        # ensure nodes exist in household base
        all_households = set(households_df['geohash8'])
        decided_nodes = decided_nodes & all_households

        # remaining households (non-deciders)
        non_decided_nodes = all_households - decided_nodes

        if not decided_nodes or not non_decided_nodes:
            continue

        # ----------------------------
        # Compute network metrics
        # ----------------------------
        deg = dict(G.degree())
        avg_neighbor_degree = nx.average_neighbor_degree(G)
        betw = nx.betweenness_centrality(G, normalized=True)

        def summarize(nodes, label):
            # only consider nodes that exist in the graph
            sub_nodes = [n for n in nodes if n in deg]
            if not sub_nodes:
                return None
            return {
                'date': date,
                'group': label,
                'n_households': len(sub_nodes),
                'avg_degree': np.mean([deg[n] for n in sub_nodes]),
                'avg_neighbor_degree': np.mean([avg_neighbor_degree[n] for n in sub_nodes]),
                'avg_betweenness': np.mean([betw[n] for n in sub_nodes]),
                'density': nx.density(G.subgraph(sub_nodes)) if len(sub_nodes) > 1 else 0
            }

        records.append(summarize(decided_nodes, 'decided'))
        records.append(summarize(non_decided_nodes, 'non_decided'))

    # ----------------------------
    # Combine results
    # ----------------------------
    result = pd.DataFrame([r for r in records if r])
    print(
        result.groupby('group')[['avg_degree', 'avg_neighbor_degree',
                                 'avg_betweenness', 'density']].mean()
    )
    return result