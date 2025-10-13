import pandas as pd
import geohash
import pdb
import numpy as np
import glob
from dateutil.relativedelta import relativedelta
import os
import networkx as nx


def load_decision_data(bbox=(-82.0, 26.48, -81.92, 26.52)):
    minx, miny, maxx, maxy = bbox

    repair = pd.read_csv(
        'decision_data/repair_coords_mapped_to_sales.csv',
        parse_dates=['record_date']
    )
    sales = pd.read_csv(
        'decision_data/sales_data_all.csv',
        parse_dates=['first_sale_in_period'],
        dayfirst=False, infer_datetime_format=True
    )

    # bbox filter
    repair = repair[
        (repair['original_lon'].between(minx, maxx)) &
        (repair['original_lat'].between(miny, maxy))
    ]
    sales = sales[
        (sales['lon'].between(minx, maxx)) &
        (sales['lat'].between(miny, maxy))
    ]

    # encode geohash8
    repair['geohash8'] = repair.apply(
        lambda r: geohash.encode(r['original_lat'], r['original_lon'], precision=8), axis=1
    )
    sales['geohash8'] = sales.apply(
        lambda r: geohash.encode(r['lat'], r['lon'], precision=8), axis=1
    )

    repair = repair[['geohash8', 'record_date']].rename(columns={'record_date': 'decision_date'})
    sales  = sales [['geohash8', 'first_sale_in_period']].rename(columns={'first_sale_in_period': 'decision_date'})

    repair['decision_type'] = 'repair'
    sales['decision_type']  = 'sales'

    decision_df = pd.concat([repair, sales], ignore_index=True)
    decision_df = decision_df.dropna(subset=['decision_date'])
    decision_df['decision_date'] = pd.to_datetime(decision_df['decision_date'], errors='coerce')
    decision_df = decision_df.dropna(subset=['decision_date'])

    print(f"[INFO] decision records kept in bbox {bbox}: {len(decision_df)}")
    return decision_df


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


def analyze_decision_network_strength(decision_df, network_dict, month_window=1):

    records = []


    for date, net_df in network_dict.items():

        G = nx.Graph()
        for _, r in net_df.iterrows():
            G.add_edge(r['group_1'], r['group_2'], weight=1, type=r['type'])


        month_start = pd.Timestamp(date)
        month_end = month_start + pd.DateOffset(months=month_window)
        current_decisions = decision_df[
            (decision_df['decision_date'] >= month_start) &
            (decision_df['decision_date'] < month_end)
        ]

        decided_nodes = set(current_decisions['geohash8'])
        all_nodes = set(G.nodes())

        non_decided_nodes = all_nodes - decided_nodes
        if not decided_nodes or not non_decided_nodes:
            continue

        deg = dict(G.degree())
        avg_neighbor_degree = nx.average_neighbor_degree(G)
        betw = nx.betweenness_centrality(G, normalized=True)

        def summarize(nodes, label):
            sub = [n for n in nodes if n in deg]
            if not sub:
                return None
            return {
                'date': date,
                'group': label,
                'n_households': len(sub),
                'avg_degree': np.mean([deg[n] for n in sub]),
                'avg_neighbor_degree': np.mean([avg_neighbor_degree[n] for n in sub]),
                'avg_betweenness': np.mean([betw[n] for n in sub]),
                'density': nx.density(G.subgraph(sub)) if len(sub) > 1 else 0
            }

        records.append(summarize(decided_nodes, 'decided'))
        records.append(summarize(non_decided_nodes, 'non_decided'))

    result = pd.DataFrame([r for r in records if r])
    print(result.groupby('group')[['avg_degree', 'avg_neighbor_degree', 'avg_betweenness', 'density']].mean())
    return result
