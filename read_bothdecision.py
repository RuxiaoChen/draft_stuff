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
import pandas as pd
import geohash
import glob, os, numpy as np


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

from dateutil.relativedelta import relativedelta
import pandas as pd

def analyze_diffusion(decision_df, network_dict, exposure_window=1, follow_window=3):

    records = []


    decision_df = decision_df.sort_values("decision_date")

    for net_date, net_df in network_dict.items():

        neighbor_map = {}
        for _, row in net_df.iterrows():
            a, b, conn_type = row["group_1"], row["group_2"], row["type"]
            neighbor_map.setdefault(a, []).append((b, conn_type))
            neighbor_map.setdefault(b, []).append((a, conn_type))

        exposure_start = net_date - relativedelta(months=exposure_window)
        follow_end = net_date + relativedelta(months=follow_window)

        recent_decisions = decision_df[
            (decision_df["decision_date"] >= exposure_start) &
            (decision_df["decision_date"] <= net_date)
        ]


        future_decisions = decision_df[
            (decision_df["decision_date"] > net_date) &
            (decision_df["decision_date"] <= follow_end)
        ]

        all_households = set(decision_df["geohash8"])
        decision_types = decision_df["decision_type"].unique()

        for h in all_households:
            if h not in neighbor_map:
                continue
            neighbors = [n for n, _ in neighbor_map[h]]
            conn_types = [t for _, t in neighbor_map[h]]

            for dtype in decision_types:

                exposed_neighbors = recent_decisions[
                    (recent_decisions["geohash8"].isin(neighbors)) &
                    (recent_decisions["decision_type"] == dtype)
                ]
                exposed = len(exposed_neighbors) > 0

                followed = len(future_decisions[
                    (future_decisions["geohash8"] == h) &
                    (future_decisions["decision_type"] == dtype)
                ]) > 0

                if exposed or followed: 
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
