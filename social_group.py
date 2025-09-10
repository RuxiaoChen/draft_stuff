import os
import geohash
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
import numpy as np

def identify_group_locations(df, min_appearances=90, night_start_hour=23, night_end_hour=5,
                            min_group_visits=10, location_precision=4, w1=0.6, w2=0.4):
    """
    Identify group locations for each device using (primarily) night-time visits.
    If a device has *no* night-time data, its day-time data are used instead.
    """
    result_df = df.copy()

    # --- 1. Preprocessing ----------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp_2']):
        result_df['timestamp_2'] = pd.to_datetime(result_df['timestamp_2'])

    # Keep only devices with sufficient appearances
    device_counts = result_df['device_id'].value_counts()
    valid_devices = device_counts[device_counts >= min_appearances].index
    result_df = result_df[result_df['device_id'].isin(valid_devices)]

    # Round lat/lon to given precision as spatial cell
    result_df['location_cell'] = (
        result_df['latitude_2'].round(location_precision).astype(str) + '_' +
        result_df['longitude_2'].round(location_precision).astype(str)
    )

    # Extract hour and date from timestamp
    result_df['hour'] = result_df['timestamp_2'].dt.hour
    result_df['date'] = result_df['timestamp_2'].dt.date

    # --- 2. Keep only night-time visits --------------------------------------------
    if night_start_hour > night_end_hour:  # Crosses midnight
        night_mask = (result_df['hour'] >= night_start_hour) | (result_df['hour'] < night_end_hour)
    else:  # Does not cross midnight
        night_mask = (result_df['hour'] >= night_start_hour) & (result_df['hour'] < night_end_hour)

    night_visits = result_df[night_mask].copy()

    # Find devices without any night-time visits → use their daytime data as fallback
    all_devices = set(result_df['device_id'].unique())
    night_devices = set(night_visits['device_id'].unique())
    devices_no_night = all_devices - night_devices


    if devices_no_night:
        backup_visits = result_df[result_df['device_id'].isin(devices_no_night)].copy()
        night_visits = pd.concat([night_visits, backup_visits], ignore_index=True)

    # --- 3. Count visits per location per device -----------------------------------
    visit_counts = (
        night_visits
        .groupby(['device_id', 'location_cell'])
        .size()
        .reset_index(name='visits')
    )

    group_candidates = visit_counts[visit_counts['visits'] >= min_group_visits].copy()

    # If a device has no location reaching the threshold → choose most visited location
    missing_dev = all_devices - set(group_candidates['device_id'].unique())
    if missing_dev:
        fallback = (
            visit_counts[visit_counts['device_id'].isin(missing_dev)]
            .sort_values(['device_id', 'visits'], ascending=[True, False])
            .groupby('device_id')
            .head(1)
        )
        group_candidates = pd.concat([group_candidates, fallback], ignore_index=True)

    # --- 4. Compute temporal consistency -------------------------------------------
    distinct_days = (
        night_visits
        .groupby(['device_id', 'location_cell'])['date']
        .nunique()
        .reset_index(name='distinct_days')
    )
    total_days = max((result_df['date'].max() - result_df['date'].min()).days + 1, 1)

    group_candidates = group_candidates.merge(distinct_days, on=['device_id', 'location_cell'], how='left')
    group_candidates['distinct_days'] = group_candidates['distinct_days'].fillna(1)
    group_candidates['temporal_consistency'] = group_candidates['distinct_days'] / total_days

    # --- 5. Compute weighted score (penalize daytime-only data) --------------------
    penalty_mask = group_candidates['device_id'].isin(devices_no_night)

    group_candidates['score'] = np.where(
        penalty_mask,
        w1 * group_candidates['visits'] * 0.5 + w2 * group_candidates['temporal_consistency'] * 0.5,
        w1 * group_candidates['visits']       + w2 * group_candidates['temporal_consistency']
    )

    # --- 6. Select the best-scored location for each device ------------------------
    best_locations = group_candidates.loc[
        group_candidates.groupby('device_id')['score'].idxmax()
    ][['device_id', 'location_cell', 'score']]

    # Retrieve original latitude and longitude of the selected location
    loc_samples = (
        night_visits
        .groupby(['device_id', 'location_cell'])
        .agg({'latitude_2': 'first', 'longitude_2': 'first'})
        .reset_index()
    )

    group_locations = best_locations.merge(loc_samples, on=['device_id', 'location_cell'])
    group_locations = group_locations.rename(
        columns={'latitude_2': 'group_latitude', 'longitude_2': 'group_longitude'}
    )[['device_id', 'group_latitude', 'group_longitude']]

    # --- 7. Merge group locations back to the original dataframe ---------------------
    result_df = result_df.merge(group_locations, on='device_id', how='left')
    result_df = result_df.drop(columns=['hour', 'date', 'location_cell'])

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
        # 2. If first 6 chars match, keep previous 8-char GeoHash
        # -------------------------------------------------
        use_prev_mask = (
            merged['prev_geohash_8'].notna() &
            (merged['group_geohash_8'].str[:6] == merged['prev_geohash_8'].str[:6])
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