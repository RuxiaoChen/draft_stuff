import pandas as pd
import geohash
import pdb

# repair_df['geohash8'] = repair_df.apply(
#     lambda row: geohash.encode(row['original_lat'], row['original_lon'], precision=8), axis=1
# )

# sales_df['geohash8'] = sales_df.apply(
#     lambda row: geohash.encode(row['lat'], row['lon'], precision=8), axis=1
# )
import pandas as pd
import glob
import geohash
from dateutil.relativedelta import relativedelta
import os
import re

# ==========================================================
# 1. Load repair/sales decision data and add geohash8 (with bbox filter)
# ==========================================================
def load_decision_data(bbox=(-82.0, 26.48, -81.92, 26.52)):
    minx, miny, maxx, maxy = bbox

    repair = pd.read_csv('decision_data/repair_coords_mapped_to_sales.csv', parse_dates=['record_date'])
    sales = pd.read_csv('decision_data/sales_data_all.csv', parse_dates=['first_sale_in_period'], dayfirst=False, infer_datetime_format=True)

    # -------- bbox filter --------
    repair = repair[
        (repair['original_lon'] >= minx) & (repair['original_lon'] <= maxx) &
        (repair['original_lat'] >= miny) & (repair['original_lat'] <= maxy)
    ]
    sales = sales[
        (sales['lon'] >= minx) & (sales['lon'] <= maxx) &
        (sales['lat'] >= miny) & (sales['lat'] <= maxy)
    ]

    # -------- encode geohash8 --------
    repair['geohash8'] = repair.apply(
        lambda r: geohash.encode(r['original_lat'], r['original_lon'], precision=8), axis=1
    )
    sales['geohash8'] = sales.apply(
        lambda r: geohash.encode(r['lat'], r['lon'], precision=8), axis=1
    )

    repair = repair[['geohash8', 'record_date']].rename(columns={'record_date': 'decision_date'})
    sales = sales[['geohash8', 'first_sale_in_period']].rename(columns={'first_sale_in_period': 'decision_date'})

    repair['decision_type'] = 'repair'
    sales['decision_type'] = 'sales'

    decision_df = pd.concat([repair, sales], ignore_index=True)
    decision_df = decision_df.dropna(subset=['decision_date'])
    decision_df['decision_date'] = pd.to_datetime(
        decision_df['decision_date'], errors='coerce', infer_datetime_format=True
    )
    decision_df = decision_df.dropna(subset=['decision_date'])

    print(f"[INFO] Filtered decision data: {len(decision_df)} records within bbox {bbox}")
    return decision_df


# ==========================================================
# 2. Load multiple monthly household social networks (with bbox filter)
# ==========================================================
def load_network_files(folder='social_network', bbox=(-82.0, 26.48, -81.92, 26.52)):
    minx, miny, maxx, maxy = bbox

    files = sorted(glob.glob(os.path.join(folder, 'Group_social_network_*.csv')))
    network_dict = {}

    for f in files:
        date = pd.to_datetime(f.split('_')[-1].split('.')[0])  # extract date
        df = pd.read_csv(f)[['group_1', 'group_2', 'type']]

        # decode geohash to (lat, lon) for both groups
        df['g1_lat'], df['g1_lon'] = zip(*df['group_1'].map(geohash.decode))
        df['g2_lat'], df['g2_lon'] = zip(*df['group_2'].map(geohash.decode))

        # bbox filtering: keep only pairs where both groups are inside the bounding box
        df = df[
            (df['g1_lon'] >= minx) & (df['g1_lon'] <= maxx) &
            (df['g1_lat'] >= miny) & (df['g1_lat'] <= maxy) &
            (df['g2_lon'] >= minx) & (df['g2_lon'] <= maxx) &
            (df['g2_lat'] >= miny) & (df['g2_lat'] <= maxy)
        ]

        # remove temporary lat/lon columns
        df = df[['group_1', 'group_2', 'type']]
        network_dict[date] = df

        print(f"[INFO] {f}: {len(df)} edges kept within bbox {bbox}")

    return network_dict

# ==========================================================
# 3. Analyze diffusion effect: decision → neighbor decision lag
# ==========================================================
def analyze_diffusion(decision_df, network_dict, month_window=3):
    """
    month_window: how many following months to observe neighbor decisions
    """
    records = []

    # Iterate through each monthly network in chronological order
    for net_date, net_df in network_dict.items():
        next_month = net_date + relativedelta(months=month_window)

        # All decisions made before the current month
        past_decisions = decision_df[decision_df['decision_date'] <= net_date]

        # Decisions that occurred within the observation window
        future_decisions = decision_df[
            (decision_df['decision_date'] > net_date) &
            (decision_df['decision_date'] <= next_month)
        ]

        # Build neighbor mapping
        neighbor_map = {}
        for _, row in net_df.iterrows():
            a, b = row['group_1'], row['group_2']
            neighbor_map.setdefault(a, set()).add(b)
            neighbor_map.setdefault(b, set()).add(a)

        # For each past decision, check if neighbors followed in the future
        for _, row in past_decisions.iterrows():
            household = row['geohash8']
            dtype = row['decision_type']
            decision_time = row['decision_date']

            if household not in neighbor_map:
                continue

            neighbors = neighbor_map[household]
            # Check if any neighbor made the same decision within the window
            neighbors_future = future_decisions[
                (future_decisions['geohash8'].isin(neighbors)) &
                (future_decisions['decision_type'] == dtype)
            ]

            if len(neighbors_future) > 0:
                records.append({
                    'anchor_household': household,
                    'decision_type': dtype,
                    'anchor_date': decision_time,
                    'network_date': net_date,
                    'num_neighbors_followed': len(neighbors_future),
                    'window_months': month_window
                })

    result = pd.DataFrame(records)
    return result


