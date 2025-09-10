import pandas as pd
import itertools
from tqdm import tqdm   # pip install tqdm
from collections import defaultdict
from scipy.stats import poisson
from statsmodels.stats.multitest import multipletests
import numpy as np

def filter_network_eadm(df_dtw_results, linked_df, start_date):
    graph_df = df_dtw_results[df_dtw_results['prediction'] == 1]

    # Convert negative DTW cost to cost and compute similarity
    graph_df['score'] = graph_df['score']
    max_cost = graph_df['score'].max()
    graph_df['similarity'] = max_cost - graph_df['score']

    # EADM Backbone Extraction Method
    time_graph = graph_df[['user_u', 'user_v']]
    time_alldf = linked_df[['device_id', 'latitude_2', 'longitude_2', 'geohash_2', 'timestamp_2']]

    # ---------- Preprocessing ----------
    time_alldf["timestamp_2"] = pd.to_datetime(time_alldf["timestamp_2"])
    time_alldf["hour_bin"] = time_alldf["timestamp_2"].dt.floor("H")

    ids_in_graph = set(time_graph["user_u"]) | set(time_graph["user_v"])
    loc_sub = time_alldf[time_alldf["device_id"].isin(ids_in_graph)]

    # For the same (geohash, hour, device), keep only the earliest time → avoid multiple rows per person
    loc_sub = (
        loc_sub
        .groupby(["geohash_2", "hour_bin", "device_id"], as_index=False)
        ["timestamp_2"].min()
    )

    edge_set = {tuple(sorted(x)) for x in time_graph[["user_u", "user_v"]].to_numpy()}

    # ---------- Scanning ----------
    records = []
    grouped = loc_sub.groupby(["geohash_2", "hour_bin"], sort=False)

    for (gh, hr), grp in tqdm(grouped, total=len(grouped), desc="Scanning groups"):
        ids = grp["device_id"].tolist()
        ts = dict(zip(ids, grp["timestamp_2"]))        # Used to access timestamp
        # Pairwise combinations within the group, much fewer than total number of edges
        for u, v in itertools.combinations(ids, 2):
            pair = tuple(sorted((u, v)))
            if pair in edge_set:
                records.append({
                    "user_u": pair[0],
                    "user_v": pair[1],
                    "time_encounter": max(ts[u], ts[v])
                })

    encounter_df = pd.DataFrame(records)

    # ------------------------------------------------------------
    # Main function: extract_backbone_eadm
    # ------------------------------------------------------------
    def extract_backbone_eadm(df,
                            time_col='time_encounter',
                            src_col='user_u',
                            dst_col='user_v',
                            resample='1min',       # Granularity for discretizing the original timeline
                            alpha=0.01,            # Global significance threshold
                            partition='bb',        # 'bb' | 'equal'
                            n_intervals=10):       # Number of intervals when partition='equal'
        """
        Implements EADM from Nadini et al. (2020) and returns the backbone network.
        """
        # ----- 0. Preprocessing -----
        df = df[[src_col, dst_col, time_col]].copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)

        # Only count one edge for the same node pair at the same time
        df['time_bin'] = df[time_col].dt.floor(resample)
        df = df.drop_duplicates([src_col, dst_col, 'time_bin'])

        # ----- 1. Determine interval boundaries Δ -----
        if partition == 'bb':
            try:
                from astropy.stats import bayesian_blocks
                # Map each edge event to a timestamp in seconds
                event_ts = (df['time_bin'] - df['time_bin'].min()).dt.total_seconds().values
                edges = bayesian_blocks(event_ts, fitness='events')
                edges = edges + df['time_bin'].min().timestamp()      # Convert to absolute seconds
                edge_ts = pd.to_datetime(edges, unit='s')
            except ImportError:
                print('Astropy not installed. Falling back to equal partitioning.')
                partition = 'equal'

        if partition == 'equal':
            t0, t1 = df['time_bin'].min(), df['time_bin'].max()
            edge_ts = pd.date_range(t0, t1, periods=n_intervals+1)

        # Assign interval labels to each record
        df['interval'] = pd.cut(df['time_bin'], bins=edge_ts,
                                right=False, labels=False, include_lowest=True)

        # ----- 2. Count edge weights within each interval -----
        weights = (
            df.groupby(['interval', src_col, dst_col])
            .size()
            .reset_index(name='w')
        )

        # ----- 3. Compute s_i(Δ) and W(Δ) -----
        s = defaultdict(lambda: defaultdict(int))   # s[Δ][node]
        W = defaultdict(int)                       # W[Δ]
        for _, r in weights.iterrows():
            Δ = int(r['interval']); u = r[src_col]; v = r[dst_col]; w = r['w']
            s[Δ][u] += w
            s[Δ][v] += w
            W[Δ]    += w

        # Length of each Δ (in minutes) — unused in Poisson but kept for consistency
        τ = {Δ: (edge_ts[Δ+1] - edge_ts[Δ]).total_seconds()/60
            for Δ in range(len(edge_ts)-1)}

        # ----- 4. Expected edge weight λ_ij = Σ_Δ a_i(Δ) a_j(Δ) -----
        λ = defaultdict(float)
        for Δ in s:
            if W[Δ] < 1:                # Skip empty intervals
                continue
            denom = 2 * W[Δ] - 1
            # Precompute all a_i values in the interval
            a = {i: s[Δ][i] / np.sqrt(denom) for i in s[Δ]}
            nodes = list(a.keys())
            for i_idx in range(len(nodes)):
                for j_idx in range(i_idx+1, len(nodes)):
                    i = nodes[i_idx]; j = nodes[j_idx]
                    λ[(i, j)] += a[i] * a[j]          # Accumulate total expectation

        # ----- 5. Observed edge weight w_ij -----
        w_obs = (weights
                .groupby([src_col, dst_col])['w']
                .sum()
                .to_dict())
        # Ensure key order (i < j)
        w_obs = {tuple(sorted(k)): v for k, v in w_obs.items()}

        # ----- 6. Compute p-value (Poisson) -----
        pairs, w_vals, lam_vals = [], [], []
        for pair, w in w_obs.items():
            lam = λ.get(pair, 0.0)
            pairs.append(pair); w_vals.append(w); lam_vals.append(lam)

        p_raw = [1 - poisson.cdf(w-1, lam) if lam > 0 else 0.0
                for w, lam in zip(w_vals, lam_vals)]

        # ----- 7. Multiple testing correction (Bonferroni) -----
        reject, p_adj, _, _ = multipletests(p_raw,
                                            alpha=alpha,
                                            method='bonferroni')

        # ----- 8. Output backbone network -----
        backbone = [(u, v, w, p)
                    for keep, (u, v), w, p in zip(reject, pairs, w_vals, p_adj)
                    if keep]

        backbone_df = pd.DataFrame(backbone,
                                columns=[src_col, dst_col,
                                            'weight', 'p_adj'])
        return backbone_df

    eadm_bb_df = extract_backbone_eadm(encounter_df,
                                    resample='1min',
                                    partition='equal',
                                    n_intervals=4)   # Use 'equal' for small datasets

    # Construct bonding links based on extracted backbone network

    # Create a set for fast lookup to check if an edge exists in the EADM backbone
    eadm_edges = set(tuple(sorted(edge)) for edge in eadm_bb_df[['user_u', 'user_v']].values)

    # For each edge in graph_df, check whether it appears in eadm_edges
    graph_df['connection type'] = graph_df.apply(
        lambda row: int(tuple(sorted((row['user_u'], row['user_v']))) in eadm_edges),
        axis=1
    )
    graph_df.to_csv(f'results/final_individual_social_network{start_date.date()}.csv',index=False)
    return graph_df