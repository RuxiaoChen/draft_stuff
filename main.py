# !pip install python-geohash
# !pip install pygeohash
# !pip install fastdtw
import imp
import pdb
from datetime import datetime
import pandas as pd

# Import our dependency
from read_data import *
from social_group import *
from dtw_fun import *
from network_filtering import *
from group_network import *
import warnings
warnings.filterwarnings("ignore")

# Data path
base_path = 'toy_datasets/'

# Define the starting month of the whole project. 2022-08-01 to 2023-07-31
# original_date=datetime(2022, 8, 1)

# Define the pre disaster date range
start_date = datetime(2020, 1, 1)
end_date = datetime(2020, 2, 1)

# 1. Filter Holiday records from 2019-12-24 18:00 to 22:00
start_time = pd.Timestamp("2022-12-24 18:00:00")
end_time   = pd.Timestamp("2022-12-24 22:00:00")

# Select the location you want
# somewhere in Maryland, Florida, NYC, Longisland, A small county in the middle of Long Island, Region around Brookhaven and Mastic Beach
selected_region = 'maryland'  

'''
1. Read Data
'''
# read_save_all_data(start_date, end_date, base_path)

filtered_df, min_lat, max_lat, min_lon, max_lon=read_region_data(selected_region, base_path)


'''
2. Filter local resident and residents with minimmun appearance in month
'''

linked_df = apply_residency_filter_appear(filtered_df, min_appearances=20)


'''
3. Identify Social Group
'''
user_group_df, result_df_with_group=user_group_links(start_date, linked_df)

'''
4. Identify Social Links using DTW
'''
df_dtw_results=dtw_compute(result_df_with_group, start_time, end_time, min_trj_len = 300)
'''
5. Network Filtering
'''
graph_df=filter_network_eadm(df_dtw_results, linked_df, start_date)


'''
6. Social Group Network
'''
group_network_df=build_group_network(graph_df, user_group_df, start_date)
