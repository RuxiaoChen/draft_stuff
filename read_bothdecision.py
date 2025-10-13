import pandas as pd
import geohash
import pdb


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

from dateutil.relativedelta import relativedelta
import pandas as pd

def analyze_diffusion(decision_df, network_dict, exposure_window=1, follow_window=3):
    """
    exposure_window: 月份，邻居决策提前多久算暴露
    follow_window: 月份，暴露后观察多久看被影响者是否决策
    """
    records = []

    # 将决策表按时间排序
    decision_df = decision_df.sort_values("decision_date")

    for net_date, net_df in network_dict.items():
        # 当前月的网络
        neighbor_map = {}
        for _, row in net_df.iterrows():
            a, b, conn_type = row["group_1"], row["group_2"], row["type"]
            neighbor_map.setdefault(a, []).append((b, conn_type))
            neighbor_map.setdefault(b, []).append((a, conn_type))

        # 时间窗口
        exposure_start = net_date - relativedelta(months=exposure_window)
        follow_end = net_date + relativedelta(months=follow_window)

        # exposure 窗口内已经决策的 households
        recent_decisions = decision_df[
            (decision_df["decision_date"] >= exposure_start) &
            (decision_df["decision_date"] <= net_date)
        ]

        # 观察窗口内的 households
        future_decisions = decision_df[
            (decision_df["decision_date"] > net_date) &
            (decision_df["decision_date"] <= follow_end)
        ]

        all_households = set(decision_df["geohash8"])
        decision_types = decision_df["decision_type"].unique()

        # 遍历每个 household 作为潜在被影响者
        for h in all_households:
            if h not in neighbor_map:
                continue
            neighbors = [n for n, _ in neighbor_map[h]]
            conn_types = [t for _, t in neighbor_map[h]]

            for dtype in decision_types:
                # 暴露：邻居在过去 exposure_window 内做出同类决策
                exposed_neighbors = recent_decisions[
                    (recent_decisions["geohash8"].isin(neighbors)) &
                    (recent_decisions["decision_type"] == dtype)
                ]
                exposed = len(exposed_neighbors) > 0

                # 被影响：该 household 在 follow_window 内做出相同决策
                followed = len(future_decisions[
                    (future_decisions["geohash8"] == h) &
                    (future_decisions["decision_type"] == dtype)
                ]) > 0

                if exposed or followed:  # 节省存储
                    records.append({
                        "network_date": net_date,
                        "household": h,
                        "decision_type": dtype,
                        "exposed": int(exposed),
                        "followed": int(followed),
                        "num_exposed_neighbors": len(exposed_neighbors),
                        "num_total_neighbors": len(neighbors),
                        "num_bonding": conn_types.count(1),
                        "num_bridging": conn_types.count(0)
                    })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # 汇总统计
    summary = (
        df.groupby(["decision_type", "network_date", "exposed"])
          .followed.mean()
          .reset_index()
          .pivot(index=["network_date", "decision_type"], columns="exposed", values="followed")
          .rename(columns={0: "P_follow_no_exposure", 1: "P_follow_exposure"})
          .reset_index()
    )
    summary["diffusion_effect"] = summary["P_follow_exposure"] - summary["P_follow_no_exposure"]

    return df, summary
