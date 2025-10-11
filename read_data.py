#Loading Data and do filtering based on study region
import pandas as pd
from datetime import datetime, timedelta
import json
import geohash
import numpy as np
from collections import defaultdict
import os
import pdb


def read_save_all_data(start_date, end_date, base_path):
    dfs = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')

        parquet_file_path = f"{base_path}{date_str}/"

        try:
            print(f"Reading data for {date_str}...")
            df_temp = pd.read_parquet(parquet_file_path, engine='pyarrow')

            # Add a date column to keep track of the source date (optional)
            df_temp['source_date'] = date_str
            dfs.append(df_temp)
            print(f"Successfully read data with shape: {df_temp.shape}")

        except Exception as e:
            print(f"Error reading data for {date_str}: {e}")

        current_date += timedelta(days=1)

    # Check if we have any dataframes to concatenate
    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        print("\nCombined Dataframe Info:")
        print(f"Shape: {combined_df.shape}")
        print(f"Column Names: {combined_df.columns.tolist()}")
        print("\nFirst 5 rows of combined data:")
        print(combined_df.head(5))

        combined_df.to_parquet(base_path+"/combined_data.parquet")
    else:
        print("No data was read successfully.")


def read_region_data(selected_region, base_path):
    # read coordinates files
    with open('coordinates.json', 'r') as f:
        coordinates = json.load(f)

    # Select the location you want
    # somewhere in Maryland, Florida, NYC, Longisland, A small county in the middle of Long Island, Region around Brookhaven and Mastic Beach

    region_coords = coordinates[selected_region]
    min_lat = region_coords['min_lat']
    max_lat = region_coords['max_lat']
    min_lon = region_coords['min_lon']
    max_lon = region_coords['max_lon']

    # combined_df = pd.read_csv(base_path)

    parquet_file_path = base_path + "combined_data.parquet"
    combined_df = pd.read_parquet(parquet_file_path, engine='pyarrow')

    combined_df.rename(columns={'cuebiq_id': 'device_id'}, inplace=True)
    combined_df.rename(columns={'start_lat': 'latitude_1'}, inplace=True)
    combined_df.rename(columns={'start_lng': 'longitude_1'}, inplace=True)
    combined_df.rename(columns={'end_lat': 'latitude_2'}, inplace=True)
    combined_df.rename(columns={'end_lng': 'longitude_2'}, inplace=True)
    combined_df.rename(columns={'start_geohash': 'geohash_1'}, inplace=True)
    combined_df.rename(columns={'end_geohash': 'geohash_2'}, inplace=True)
    combined_df.rename(columns={'start_zoned_datetime': 'utc_timestamp_1'}, inplace=True)
    combined_df.rename(columns={'end_zoned_datetime': 'utc_timestamp_2'}, inplace=True)

    filtered_df = apply_residency_filter(combined_df, min_lat, max_lat, min_lon, max_lon)



    print(f"Original DataFrame shape: {filtered_df.shape}")


    geohash_1_list = []
    geohash_2_list = []

    for lat1, lon1, lat2, lon2 in zip(
        filtered_df["latitude_1"], filtered_df["longitude_1"],
        filtered_df["latitude_2"], filtered_df["longitude_2"]
    ):
        geohash_1_list.append(geohash.encode(lat1, lon1, precision=9))
        geohash_2_list.append(geohash.encode(lat2, lon2, precision=9))

    filtered_df["geohash_1"] = geohash_1_list
    filtered_df["geohash_2"] = geohash_2_list

    # Keep only the specified columns
    columns_to_keep = [
        'device_id',
        # 'linked_trip_id',
        'utc_timestamp_1',
        # 'utc_offset_1',
        'utc_timestamp_2',
        # 'utc_offset_2',
        'latitude_1',
        'longitude_1',
        'geohash_1',
        'latitude_2',
        'longitude_2',
        'geohash_2'
    ]
    
    filtered_df = filtered_df[columns_to_keep]

    # Create unix_time columns (adding offset to timestamps)
    filtered_df['unix_time_1'] = filtered_df['utc_timestamp_1']
    filtered_df['unix_time_2'] = filtered_df['utc_timestamp_2']
    # Generate human-readable timestamps
    def unix_to_mdy_hms(unix_time):
        return datetime.fromtimestamp(unix_time).strftime('%m/%d/%Y %H:%M:%S')
    filtered_df['timestamp_1'] = filtered_df['unix_time_1'].apply(unix_to_mdy_hms)
    filtered_df['timestamp_2'] = filtered_df['unix_time_2'].apply(unix_to_mdy_hms)

    # def iso8601_to_mdy_hms(iso_str):
    #     dt = datetime.fromisoformat(iso_str)
        # return dt.strftime('%m/%d/%Y %H:%M:%S')

    # filtered_df['timestamp_1'] = filtered_df['utc_timestamp_1'].apply(iso8601_to_mdy_hms)
    # filtered_df['timestamp_2'] = filtered_df['utc_timestamp_2'].apply(iso8601_to_mdy_hms)

    # Add original_device_id column (ground truth)
    filtered_df['original_device_id'] = filtered_df['device_id']

    # Convert timestamps to datetime for date operations
    filtered_df['datetime_1'] = pd.to_datetime(filtered_df['timestamp_1'])
    filtered_df['date_1'] = filtered_df['datetime_1'].dt.date

    # Get the total number of unique device_ids
    all_device_ids = filtered_df['device_id'].unique()
    return filtered_df, min_lat, max_lat, min_lon, max_lon

def link_device_trajectories_optimized(df, max_time_gap_seconds=3600, geohash_digit_tolerance=8):
    """
    Links device trajectories based on temporal proximity and geohash similarity.
    When a device ID disappears, finds the new device ID with the closest geohash match within the specified time window.
    Also handles the case of device ID shuffling:
      1. Before the function starts, it reads a JSON file (default: device_id_shuffle_map.json).
         If a device_id in df is found in this mapping (i.e., it's a new ID), it will be replaced with its original ID.
      2. During the linking process, if a device ID link is detected (i.e., original ID differs from new ID),
         the new-to-original ID pair will be added to the JSON mapping file.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing trajectory data with device_id, unix_time_1,
        unix_time_2, geohash_1, geohash_2
    max_time_gap_seconds : int
        Maximum allowed time gap between trajectories to be linked (in seconds)
    geohash_digit_tolerance : int
        Minimum number of matching digits required in geohash comparison

    Returns:
    --------
    pandas.DataFrame
        DataFrame with linked device IDs
    dict
        Mapping of device IDs that were linked
    """


    # --- New: Load original-to-shuffled ID mapping from JSON and replace new IDs in df ---
    shuffle_file = "device_id_shuffle_map.json"
    if os.path.exists(shuffle_file):
        with open(shuffle_file, 'r') as f:
            shuffle_mapping = json.load(f)
        print(f"Loaded existing shuffle mapping from {shuffle_file}.")
    else:
        shuffle_mapping = {}
        print(f"No existing shuffle mapping found. Starting with an empty mapping.")

    # Replace new device IDs in df with original IDs using the loaded mapping
    df['device_id'] = df['device_id'].apply(lambda x: shuffle_mapping.get(x, x))

    # -----------------------------------------------------------------------------------

    print("Starting trajectory linking process...")
    print(f"Parameters: max_time_gap={max_time_gap_seconds}s, geohash_digits={geohash_digit_tolerance}")

    linked_df = df.copy()
    linked_df = linked_df.sort_values(['device_id', 'unix_time_1'])

    device_groups = linked_df.groupby('device_id')

    device_last_points = {}
    for device_id, group in device_groups:
        last_row = group.loc[group['unix_time_2'].idxmax()]
        device_last_points[device_id] = {
            'time': last_row['unix_time_2'],
            'geohash': last_row['geohash_2'],
            'last_seen_idx': last_row.name
        }

    device_first_points = {}
    for device_id, group in device_groups:
        first_row = group.loc[group['unix_time_1'].idxmin()]
        device_first_points[device_id] = {
            'time': first_row['unix_time_1'],
            'geohash': first_row['geohash_1'],
            'first_seen_idx': first_row.name
        }

    print(f"Identified last points for {len(device_last_points)} device IDs")
    print(f"Identified first points for {len(device_first_points)} device IDs")

    geohash_similarity_cache = {}
    def geohash_similarity(geohash1, geohash2):
        cache_key = (geohash1, geohash2)
        if cache_key in geohash_similarity_cache:
            return geohash_similarity_cache[cache_key]

        min_len = min(len(geohash1), len(geohash2))
        for i in range(min_len):
            if geohash1[i] != geohash2[i]:
                result = i
                geohash_similarity_cache[cache_key] = result
                return result
        result = min_len
        geohash_similarity_cache[cache_key] = result
        return result

    time_buckets = defaultdict(list)
    for device_id, first_point in device_first_points.items():
        time_bucket = first_point['time'] // max_time_gap_seconds
        time_buckets[time_bucket].append((device_id, first_point))

    linked_device_pairs = []
    unlinked_device_count = 0
    already_linked_new_devices = set()

    for device_id, last_point in device_last_points.items():
        last_time_bucket = last_point['time'] // max_time_gap_seconds
        relevant_buckets = [last_time_bucket, last_time_bucket + 1]

        best_match = None
        best_similarity = -1

        for bucket in relevant_buckets:
            if bucket not in time_buckets:
                continue

            for new_device_id, first_point in time_buckets[bucket]:
                if (new_device_id == device_id or
                    first_point['time'] <= last_point['time'] or
                    new_device_id in already_linked_new_devices):
                    continue

                time_diff = first_point['time'] - last_point['time']
                if time_diff > max_time_gap_seconds:
                    continue

                similarity = geohash_similarity(last_point['geohash'], first_point['geohash'])
                if similarity >= geohash_digit_tolerance and similarity > best_similarity:
                    best_match = new_device_id
                    best_similarity = similarity

        if best_match:
            linked_device_pairs.append((device_id, best_match))
            already_linked_new_devices.add(best_match)
        else:
            unlinked_device_count += 1

    print(f"Found {len(linked_device_pairs)} potential device links")
    print(f"Could not find suitable matches for {unlinked_device_count} device IDs")

    device_id_mapping = {}
    def find_root_device(device_id):
        path = []
        current = device_id
        while current in device_id_mapping:
            path.append(current)
            current = device_id_mapping[current]
        for node in path:
            device_id_mapping[node] = current
        return current

    for old_id, new_id in linked_device_pairs:
        device_id_mapping[new_id] = old_id

    all_device_ids = set(linked_df['device_id'].unique())
    root_mapping = {device_id: find_root_device(device_id) for device_id in all_device_ids}

    linked_df['linked_device_id'] = linked_df['device_id'].map(root_mapping).fillna(linked_df['device_id'])

    print("Trajectory linking completed")

    # --- New: Update JSON mapping file ---
    # For all new-to-original mappings where IDs differ, update shuffle_mapping
    for new_id, original_id in device_id_mapping.items():
        if new_id != original_id and new_id not in shuffle_mapping:
            shuffle_mapping[new_id] = original_id
    with open(shuffle_file, 'w') as f:
        json.dump(shuffle_mapping, f, indent=4)
    print(f"Updated shuffle mapping written to {shuffle_file}.")
    # -----------------------------------------------------------------------------------

    return linked_df, device_id_mapping


def apply_residency_filter(df, min_lat, max_lat, min_lon, max_lon):

    # in_box = (
    #     df['latitude_2'].between(min_lat, max_lat) &
    #     df['longitude_2'].between(min_lon, max_lon)
    # )
    # total_counts   = df['device_id'].value_counts()
    # in_box_counts  = df[in_box]['device_id'].value_counts()


    # Check which start points (latitude_1, longitude_1) are in the box
    start_points_in_box = (
        df['latitude_1'].between(min_lat, max_lat) & df['longitude_1'].between(min_lon, max_lon)
    )
    
    # Check which end points (latitude_2, longitude_2) are in the box  
    end_points_in_box = (
        df['latitude_2'].between(min_lat, max_lat) & df['longitude_2'].between(min_lon, max_lon)
    )
    
    # Count start points in box for each device
    start_in_box_counts = df[start_points_in_box]['device_id'].value_counts()
    
    # Count end points in box for each device
    end_in_box_counts = df[end_points_in_box]['device_id'].value_counts()
    
    # Total in_box count is sum of start and end points in box
    in_box_counts = start_in_box_counts.add(end_in_box_counts, fill_value=0)

    # Total points per device is 2 times the number of rows (each row has start and end point)
    total_counts = df['device_id'].value_counts() * 2

    stats = (
        pd.DataFrame({
            'total':   total_counts,
            'in_box':  in_box_counts
        })
        .fillna(0)
    )

    valid_devices = stats[ stats['in_box'] >= stats['total'] / 2].index

    return df[ df['device_id'].isin(valid_devices) ].copy()

def apply_residency_filter_appear(df, min_appearances=30):
    device_counts = df['device_id'].value_counts()

    valid_devices = device_counts[device_counts >= min_appearances].index
    filtered_df = df[df['device_id'].isin(valid_devices)].copy()

    return filtered_df