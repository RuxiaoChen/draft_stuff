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

# ==========================================================
# 1. 加载 repair / sales decision 数据并加上 geohash8
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
# 2. 载入多个月的 household social network
#    文件命名如：results/Group_social_network_2023-04-01.csv
# ==========================================================
def load_network_files(folder='social_network'):
    files = sorted(glob.glob(os.path.join(folder, 'Group_social_network_*.csv')))
    network_dict = {}
    for f in files:
        date = pd.to_datetime(f.split('_')[-1].split('.')[0])  # 从文件名提取日期
        df = pd.read_csv(f)
        df = df[['group_1', 'group_2', 'type']]
        network_dict[date] = df
    return network_dict


# ==========================================================
# 3. 分析扩散效应：decision → neighbor decision lag
# ==========================================================
def analyze_diffusion(decision_df, network_dict, month_window=3):
    """
    month_window: 观察后续多少个月的邻居决策
    """
    records = []

    # 按时间顺序遍历每个月的网络
    for net_date, net_df in network_dict.items():
        next_month = net_date + relativedelta(months=month_window)

        # 当前时间点前的所有decision
        past_decisions = decision_df[decision_df['decision_date'] <= net_date]

        # 在观察窗口内发生的decision
        future_decisions = decision_df[
            (decision_df['decision_date'] > net_date) &
            (decision_df['decision_date'] <= next_month)
        ]

        # 建立邻居映射
        neighbor_map = {}
        for _, row in net_df.iterrows():
            a, b = row['group_1'], row['group_2']
            neighbor_map.setdefault(a, set()).add(b)
            neighbor_map.setdefault(b, set()).add(a)

        # 遍历每个过去的decision，看看邻居未来是否跟进
        for _, row in past_decisions.iterrows():
            household = row['geohash8']
            dtype = row['decision_type']
            decision_time = row['decision_date']

            if household not in neighbor_map:
                continue

            neighbors = neighbor_map[household]
            # 看未来窗口内是否有邻居做了相同 decision
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


# ==========================================================
# 4. 主程序
# ==========================================================
if __name__ == '__main__':
    decision_df = load_decision_data()
    network_dict = load_network_files('social_network')
    pdb.set_trace()
    result_df = analyze_diffusion(decision_df, network_dict, month_window=3)

    result_df.to_csv('results/household_diffusion_analysis.csv', index=False)
    print(result_df.head())
    print("共发现邻居扩散事件数量:", len(result_df))
