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
# 1. Load repair/sales decision data and add geohash8
# ==========================================================
def load_decision_data():
    repair = pd.read_csv('decision_data/repair_coords_mapped_to_sales.csv', parse_dates=['record_date'])
    sales = pd.read_csv('decision_data/sales_data_all.csv', parse_dates=['first_sale_in_period'], dayfirst=False, infer_datetime_format=True)

    repair['geohash8'] = repair.apply(lambda r: geohash.encode(r['original_lat'], r['original_lon'], precision=8), axis=1)
    sales['geohash8'] = sales.apply(lambda r: geohash.encode(r['lat'], r['lon'], precision=8), axis=1)

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
    return decision_df


# ==========================================================
# 2. Load monthly social network files
# ==========================================================
def load_network_files(folder='results'):
    files = sorted(glob.glob(os.path.join(folder, 'lee_county_20*.csv')))
    network_dict = {}

    for f in files:
        match = re.search(r'lee_county_(\d{6})', os.path.basename(f))
        if not match:
            continue
        date_str = match.group(1)  # e.g. '202208'
        year = int(date_str[:4])
        month = int(date_str[4:])
        date_obj = pd.Timestamp(year=year, month=month, day=1)

        df = pd.read_csv(f, dtype={'group_1': str, 'group_2': str, 'type': int})
        network_dict[date_obj] = df[['group_1', 'group_2', 'type']]
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


