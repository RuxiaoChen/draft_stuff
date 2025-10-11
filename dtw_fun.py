import geohash
import pandas as pd
import pygeohash as pgh
import numpy as np
from numba.typed import List as NbList
from itertools import combinations
import random, math
from tqdm import tqdm
from numba import njit

# Suppose your new dataset is in a DataFrame called filtered_df,
# which has columns for two check-ins per row. We only want the second one.

# 1. Rename columns to match Gowalla style
#    user_id -> device_id or original_device_id (choose one)
#    timestamp -> timestamp_2
#    lat -> latitude_2
#    lon -> longitude_2
#    We'll unify them as: user_id, timestamp, lat, lon, poi (poi = geohash_2)

# renamed_df = filtered_df.rename(columns={
#     'device_id': 'user_id',        # or 'original_device_id' if preferred
#     'timestamp_2': 'timestamp',
#     'latitude_2': 'lat',
#     'longitude_2': 'lon'
# })



def dtw_compute(result_df_with_groups,start_time, end_time, min_trj_len = 20):
    df1 = result_df_with_groups[['device_id', 'timestamp_1', 'latitude_1', 'longitude_1']].rename(columns={
        'device_id': 'user_id',
        'timestamp_1': 'timestamp',
        'latitude_1': 'lat',
        'longitude_1': 'lon'
    })

    df2 = result_df_with_groups[['device_id', 'timestamp_2', 'latitude_2', 'longitude_2']].rename(columns={
        'device_id': 'user_id',
        'timestamp_2': 'timestamp',
        'latitude_2': 'lat',
        'longitude_2': 'lon'
    })

    renamed_df = pd.concat([df1, df2], ignore_index=True).dropna(subset=['lat', 'lon', 'timestamp'])


    # 2. Convert the timestamp to the Gowalla-like format: YYYY-MM-DDTHH:MM:SSZ
    renamed_df['timestamp'] = pd.to_datetime(renamed_df['timestamp'])
    renamed_df['timestamp'] = renamed_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # 3. Compute 8-digit geohash as the new 'poi'
    renamed_df['poi'] = renamed_df.apply(lambda row: pgh.encode(row['lat'], row['lon'], precision=9), axis=1)

    # 4. Filter to keep only the relevant columns
    columns_needed = ['user_id', 'timestamp', 'lat', 'lon', 'poi']
    checkins = renamed_df[columns_needed].copy()

    '''
    Position of Interest (PoI) + Dynamic Time Warping (DTW) Type 1
    '''

    # Group by user_id and filter users with fewer than 10 check-ins
    # -------------------------------------------------
    # 0. Parameters
    # -------------------------------------------------
    topk        = 3           # Take the k smallest DTW values
    radius      = 2           # Fast dtw allows ±2 steps of alignment shift. Set this value to bigger one will increase runningtime. Original DTW doesn't have this constraint.

    # -------------------------------------------------
    # 1. Data preparation (vectorized for speed and memory efficiency)
    # -------------------------------------------------
    checkins['timestamp'] = pd.to_datetime(checkins['timestamp'])
    active_mask = checkins.groupby('user_id')['user_id'].transform('size') >= min_trj_len
    print('Number of active users:', len(checkins[active_mask]))
    checkins = checkins.loc[active_mask].copy()

    # —— 1.1  Map PoI to index & coordinates —— #
    uniq_poi = checkins.drop_duplicates('poi')[['poi', 'lat', 'lon']]
    uniq_poi['idx'] = pd.factorize(uniq_poi['poi'])[0]
    poi2idx   = dict(zip(uniq_poi['poi'], uniq_poi['idx']))
    idx2coord = uniq_poi[['lat', 'lon']].to_numpy()        # shape (N,2)
    checkins['poi_idx'] = checkins['poi'].map(poi2idx)     # int32

    # —— 1.2  Get daily trajectories (as integer sequences) —— #
    checkins['date'] = checkins['timestamp'].dt.date
    user_daily_idx_py = (
        checkins.groupby(['user_id', 'date'])['poi_idx']
                .agg(list)
                .groupby(level=0).agg(list)
                .to_dict()                                  # {uid: [[idx...], ...]}
    )

    # —— 1.3  Convert to numba typed-list (required for JIT) —— #
    def to_nb_list(days_py):
        nb_days = NbList()
        for day in days_py:
            nb_days.append(np.asarray(day, dtype=np.int32))
        return nb_days

    user_daily_nb = {uid: to_nb_list(days) for uid, days in user_daily_idx_py.items()}

    # —— 1.4  Candidate pairs (users sharing ≥1 PoI) —— #
    user_all_pois = (
        checkins.groupby('user_id')['poi_idx']
                .agg(lambda s: set(s))
                .to_dict()
    )
    inv_index = (
        checkins.groupby('poi_idx')['user_id']
                .agg(lambda s: set(s))
    )
    candidate_pairs = set()
    for users_set in inv_index:
        if len(users_set) > 1:
            candidate_pairs.update(combinations(sorted(users_set), 2))
    candidate_pairs = list(candidate_pairs)
    random.shuffle(candidate_pairs)

    # -------------------------------------------------
    # 2. Numba-JIT DTW (with Sakoe-Chiba window)
    # -------------------------------------------------
    lat_rad = np.radians(idx2coord[:, 0]).astype(np.float64)
    lon_rad = np.radians(idx2coord[:, 1]).astype(np.float64)
    R_EARTH = 6371.0

    @njit(inline='always')
    def haversine_idx(i, j, lat_r, lon_r):
        if i == j:
            return 0.0
        dlat = lat_r[j] - lat_r[i]
        dlon = lon_r[j] - lon_r[i]
        a = (math.sin(dlat * 0.5) ** 2 +
            math.cos(lat_r[i]) * math.cos(lat_r[j]) *
            math.sin(dlon * 0.5) ** 2)
        return 2.0 * R_EARTH * math.asin(math.sqrt(a))

    @njit
    def dtw_window(seq1, seq2, w, lat_r, lon_r):
        n, m = len(seq1), len(seq2)
        w    = max(w, abs(n - m))
        big  = 1e20
        dp   = np.full((n + 1, m + 1), big, dtype=np.float64)
        dp[0, 0] = 0.0

        for i in range(1, n + 1):
            j_lo = max(1, i - w)
            j_hi = min(m, i + w) + 1
            a_idx = seq1[i - 1]
            for j in range(j_lo, j_hi):
                b_idx = seq2[j - 1]
                cost  = haversine_idx(a_idx, b_idx, lat_r, lon_r)
                best  = dp[i - 1, j]
                if dp[i, j - 1] < best:
                    best = dp[i, j - 1]
                if dp[i - 1, j - 1] < best:
                    best = dp[i - 1, j - 1]
                dp[i, j] = cost + best

        # Backtrace to count path length
        i, j, path_len = n, m, 0
        while i or j:
            path_len += 1
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                a, b, c = dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]
                if a <= b and a <= c:
                    i, j = i - 1, j - 1
                elif b < c:
                    i -= 1
                else:
                    j -= 1
        return dp[n, m] / path_len

    @njit
    def aggregate_topk(trajs_u, trajs_v, k, radius, lat_r, lon_r):
        tot = len(trajs_u) * len(trajs_v)
        dists = np.empty(tot, dtype=np.float32)
        p = 0
        for i in range(len(trajs_u)):
            for j in range(len(trajs_v)):
                dists[p] = dtw_window(trajs_u[i], trajs_v[j],
                                    radius, lat_r, lon_r)
                p += 1
        k = k if k < tot else tot
        for i in range(k):
            min_idx = i
            for j in range(i + 1, tot):
                if dists[j] < dists[min_idx]:
                    min_idx = j
            dists[i], dists[min_idx] = dists[min_idx], dists[i]
        return dists[:k].mean()

    # -------------------------------------------------
    # 3. Compute DTW cost for all candidate pairs
    # -------------------------------------------------
    print("Fast-DTW running …")
    candidate_costs = []
    for (u, v) in tqdm(candidate_pairs):
        cost_uv = aggregate_topk(
            user_daily_nb[u],
            user_daily_nb[v],
            topk, radius, lat_rad, lon_rad
        )
        candidate_costs.append(cost_uv)

    all_costs = np.array(candidate_costs, dtype=np.float32)


    # 20% of the data will be positive
    desired_ratio=0.2
    # Sort the cost value
    sorted_costs = np.sort(all_costs)
    # Calculate place of the desired threshold
    threshold_index = int(len(sorted_costs) * desired_ratio)
    fixed_threshold = sorted_costs[threshold_index]
    print(f"Threshold selected: {fixed_threshold}")


    # # Plot histogram of aggregated DTW costs to observe the distribution
    # plt.figure()
    # counts, bins, patches = plt.hist(all_costs, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    # plt.xlabel('Aggregated DTW Cost')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Aggregated DTW Cost for Candidate Pairs')
    # plt.show()


    # 6. Predict social links based on threshold: candidate pairs with DTW cost < threshold are predicted as social links
    predictions = [1 if cost <= fixed_threshold else 0 for cost in all_costs]
    predictions = np.array(predictions)
    print("Number of candidate pairs predicted as social links:", np.sum(predictions))



    # Generate prediction results for candidate pairs based on the fixed threshold
    dtw_results = []
    scores=all_costs
    for i, (u, v) in enumerate(candidate_pairs):
        score = scores[i]
        prediction = 1 if score <= fixed_threshold else 0
        dtw_results.append({
            "threshold": fixed_threshold,
            "user_u": u,
            "user_v": v,
            "score": score,
            "prediction": prediction
        })

    df_dtw_results = pd.DataFrame(dtw_results)

    # ---------------------------
    # Part 2: Use Holiday Data to Compute Bonding Links
    # ---------------------------

    # Filter Holiday records
    if not pd.api.types.is_datetime64_any_dtype(result_df_with_groups['timestamp_2']):
        result_df_with_groups['timestamp_2'] = pd.to_datetime(result_df_with_groups['timestamp_2'])
        result_df_with_groups['timestamp_2'] = result_df_with_groups['timestamp_2'].dt.tz_localize(None)
    holiday_df = result_df_with_groups[(result_df_with_groups['timestamp_2'] >= start_time) & (result_df_with_groups['timestamp_2'] < end_time)]
    print("Records during holiday:", len(holiday_df))

    # Compute geometric center (average lat/lon) for each user during this time window
    geom_centers = holiday_df.groupby('device_id').agg({
        'latitude_2': 'mean',
        'longitude_2': 'mean'
    }).reset_index()

    # Compute 7-character Geohash (grid division)
    geom_centers["geohash_7"] = [
        geohash.encode(float(lat), float(lon), precision=7)
        for lat, lon in zip(geom_centers["latitude_2"], geom_centers["longitude_2"])
    ]
    print("Number of holiday bonding nodes area:", len(geom_centers))

    # Construct user pairs in the same 7-character Geohash grid (bonding candidate pairs)
    gh_to_devices = geom_centers.groupby('geohash_7')['device_id'].apply(list).to_dict()
    bonding_candidate_pairs = []
    for gh, devices in gh_to_devices.items():
        if len(devices) > 1:
            pairs = list(combinations(sorted(devices), 2))
            bonding_candidate_pairs.extend(pairs)
    print("Number of holiday bonding user pairs (edges):", len(bonding_candidate_pairs))

    bonding_set = set(bonding_candidate_pairs)

    # ---------------------------
    # Part 3: Combine DTW predictions and bonding links to determine connection type
    # ---------------------------
    # If a candidate pair is predicted as connected (prediction == 1) under the fixed DTW threshold
    # and the two users are in the same bonding grid, classify them as a bonding connection

    def determine_connection_type(row):
        pair = tuple(sorted([row["user_u"], row["user_v"]]))
        if pair in bonding_set:
            return 1
        else:
            return 0

    df_dtw_results["Holiday bonding"] = df_dtw_results.apply(determine_connection_type, axis=1)

    # Save final results to CSV
    df_dtw_results.to_csv("results/Individual_social_network.csv", index=False)
    print("DTW Prediction details saved to Individual_social_network.csv")
    return df_dtw_results



    # ============ (Optional) Plot the relationship between threshold and number of predictions ============
    # # Generate a series of thresholds covering the score range
    # min_score, max_score = scores.min(), scores.max()
    # thresholds = np.linspace(min_score, max_score, num=20)


    # positives = [np.sum(scores <= t) for t in thresholds]
    # negatives = [len(scores) - np.sum(scores <= t) for t in thresholds]

    # plt.figure()
    # plt.plot(thresholds, positives, label='Predicted Positive')
    # plt.plot(thresholds, negatives, label='Predicted Negative')
    # plt.xlabel('Threshold (Score)')
    # plt.ylabel('Number of Candidate Pairs')
    # plt.title('Prediction Counts at Different Thresholds')
    # plt.legend()
    # plt.show()
