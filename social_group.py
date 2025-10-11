import os
import geohash
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
import numpy as np
import pdb



def identify_group_locations(df, min_appearances=20, night_start_hour=23, night_end_hour=5,
                             min_group_visits=5, location_precision=4, w1=0.6, w2=0.4):
    """
    Identify each device's main residence/group location using both *_1 and *_2 data.
    If a device has no night-time data, use daytime data as fallback.
    """
    result_df = df.copy()

    # --- 1. Combine *_1 and *_2 data ------------------------------------------------
    df1 = result_df[['device_id', 'timestamp_1', 'latitude_1', 'longitude_1']].rename(
        columns={'timestamp_1': 'timestamp', 'latitude_1': 'latitude', 'longitude_1': 'longitude'}
    )
    df2 = result_df[['device_id', 'timestamp_2', 'latitude_2', 'longitude_2']].rename(
        columns={'timestamp_2': 'timestamp', 'latitude_2': 'latitude', 'longitude_2': 'longitude'}
    )

    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df = combined_df.dropna(subset=['latitude', 'longitude', 'timestamp'])

    # --- 2. Time conversion and filtering -------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(combined_df['timestamp']):
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

    # Keep only devices with enough total appearances
    device_counts = combined_df['device_id'].value_counts()
    valid_devices = device_counts[device_counts >= min_appearances].index
    combined_df = combined_df[combined_df['device_id'].isin(valid_devices)]

    # Round coordinates into spatial cells
    combined_df['location_cell'] = (
        combined_df['latitude'].round(location_precision).astype(str) + '_' +
        combined_df['longitude'].round(location_precision).astype(str)
    )

    # Extract hour and date
    combined_df['hour'] = combined_df['timestamp'].dt.hour
    combined_df['date'] = combined_df['timestamp'].dt.date

    # --- 3. Select night-time visits (or fallback to daytime) -----------------------
    if night_start_hour > night_end_hour:  # crosses midnight
        night_mask = (combined_df['hour'] >= night_start_hour) | (combined_df['hour'] < night_end_hour)
    else:
        night_mask = (combined_df['hour'] >= night_start_hour) & (combined_df['hour'] < night_end_hour)

    night_visits = combined_df[night_mask].copy()

    # fallback for devices with no night data
    all_devices = set(combined_df['device_id'].unique())
    night_devices = set(night_visits['device_id'].unique())
    devices_no_night = all_devices - night_devices

    if devices_no_night:
        backup_visits = combined_df[combined_df['device_id'].isin(devices_no_night)].copy()
        night_visits = pd.concat([night_visits, backup_visits], ignore_index=True)

    # --- 4. Count visits per (device, location_cell) -------------------------------
    visit_counts = (
        night_visits
        .groupby(['device_id', 'location_cell'])
        .size()
        .reset_index(name='visits')
    )

    group_candidates = visit_counts[visit_counts['visits'] >= min_group_visits].copy()

    # fallback: most visited location
    missing_dev = all_devices - set(group_candidates['device_id'].unique())
    if missing_dev:
        fallback = (
            visit_counts[visit_counts['device_id'].isin(missing_dev)]
            .sort_values(['device_id', 'visits'], ascending=[True, False])
            .groupby('device_id')
            .head(1)
        )
        group_candidates = pd.concat([group_candidates, fallback], ignore_index=True)

    # --- 5. Temporal consistency ----------------------------------------------------
    distinct_days = (
        night_visits
        .groupby(['device_id', 'location_cell'])['date']
        .nunique()
        .reset_index(name='distinct_days')
    )
    total_days = max((combined_df['date'].max() - combined_df['date'].min()).days + 1, 1)

    group_candidates = group_candidates.merge(distinct_days, on=['device_id', 'location_cell'], how='left')
    group_candidates['distinct_days'] = group_candidates['distinct_days'].fillna(1)
    group_candidates['temporal_consistency'] = group_candidates['distinct_days'] / total_days

    # --- 6. Weighted score ----------------------------------------------------------
    penalty_mask = group_candidates['device_id'].isin(devices_no_night)
    group_candidates['score'] = np.where(
        penalty_mask,
        w1 * group_candidates['visits'] * 0.5 + w2 * group_candidates['temporal_consistency'] * 0.5,
        w1 * group_candidates['visits']       + w2 * group_candidates['temporal_consistency']
    )

    # --- 7. Select best-scored location ---------------------------------------------
    best_locations = group_candidates.loc[
        group_candidates.groupby('device_id')['score'].idxmax()
    ][['device_id', 'location_cell', 'score']]

    # retrieve latitude, longitude
    loc_samples = (
        night_visits
        .groupby(['device_id', 'location_cell'])
        .agg({'latitude': 'first', 'longitude': 'first'})
        .reset_index()
    )

    group_locations = best_locations.merge(loc_samples, on=['device_id', 'location_cell'])
    group_locations = group_locations.rename(
        columns={'latitude': 'group_latitude', 'longitude': 'group_longitude'}
    )[['device_id', 'group_latitude', 'group_longitude']]

    # --- 8. Merge back to original dataframe ---------------------------------------
    result_df = result_df.merge(group_locations, on='device_id', how='left')
    return result_df



def user_group_links(start_date, linked_df):
    result_df_with_groups = identify_group_locations(linked_df)
    # Assume your DataFrame is named result_df_with_groups
    # It contains the following columns:
    # device_id, linked_trip_id, latitude, longitude, timestamp, group_latitude, group_longitude

    # 1. Encode group_latitude and group_longitude into a 9-character Geohash
    result_df_with_groups["group_geohash_8"] = result_df_with_groups.apply(
        lambda row: geohash.encode(row["group_latitude"], row["group_longitude"], precision=8),
        axis=1
    )

    # 2. Decode the 9-character Geohash to the center point (lat, lon) of the grid
    #    decode() returns a tuple (center_lat, center_lon)
    result_df_with_groups["group_center"] = result_df_with_groups["group_geohash_8"].apply(geohash.decode)

    # 3. Split the tuple into two columns: group_latitude_8, group_longitude_8
    result_df_with_groups["group_latitude_8"] = result_df_with_groups["group_center"].apply(lambda x: x[0])
    result_df_with_groups["group_longitude_8"] = result_df_with_groups["group_center"].apply(lambda x: x[1])

    # 4. If the intermediate column group_center is no longer needed, delete it
    result_df_with_groups.drop(columns=["group_center"], inplace=True)

    # Select the columns to retain
    cols_to_keep = ['device_id', 'group_latitude', 'group_longitude',
                    'group_geohash_8', 'group_latitude_8', 'group_longitude_8']

    # Extract these columns and remove duplicate device_id entries
    result_df_subset = result_df_with_groups[cols_to_keep].copy().drop_duplicates(subset=['device_id'])

    # group_latitude_8, group_longitude_8 are center coordinate of the grid cell by geohash, group_latitude, group_longitude are orginal group coordinate
    # View the processed data
    print(result_df_with_groups.head())
    print("Number of group count:", result_df_subset['device_id'].nunique())


    last_month = start_date - relativedelta(months=1)
    os.makedirs('results', exist_ok=True) 
    prev_path  = Path(f"results/user_group_relation_{last_month.date()}.csv")

    if prev_path.exists():
        prev_df = pd.read_csv(prev_path, dtype={'device_id': str, 'group_geohash_8': str})
        prev_df = prev_df[['device_id', 'group_geohash_8']].rename(columns={'group_geohash_8': 'prev_geohash_8'})

        # -------------------------------------------------
        # 1. Merge current and previous GeoHashes by device_id
        # -------------------------------------------------
        merged = result_df_subset.merge(prev_df, on='device_id', how='left')

        # -------------------------------------------------
        # 2. If first 7 chars match, keep previous 8-char GeoHash
        # -------------------------------------------------
        use_prev_mask = (
            merged['prev_geohash_8'].notna() &
            (merged['group_geohash_8'].str[:7] == merged['prev_geohash_8'].str[:7])
        )
        merged.loc[use_prev_mask, 'group_geohash_8'] = merged.loc[use_prev_mask, 'prev_geohash_8']

        # -------------------------------------------------
        # 3. Recompute centre lat/lon for any rows that just changed
        # -------------------------------------------------
        changed = merged['group_latitude_8'].isna() | use_prev_mask
        merged.loc[changed, ['group_latitude_8', 'group_longitude_8']] = (
            merged.loc[changed, 'group_geohash_8']
                .apply(lambda gh: pd.Series(geohash.decode(gh)))
                .values
        )

        # drop helper column & write out
        result_df_subset = merged.drop(columns='prev_geohash_8')
    else:
        print(f"[WARN] Previous-month file not found: {prev_path}")

    # -------------------------------------------------
    # 4. Save the updated mapping for this month
    # -------------------------------------------------

    result_df_subset.to_csv(f"results/user_group_relation_{start_date.date()}.csv", index=False)
    return result_df_subset, result_df_with_groups