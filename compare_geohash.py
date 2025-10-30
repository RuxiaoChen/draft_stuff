import pandas as pd
import numpy as np
from geopy.distance import geodesic,great_circle
import geohash
from tqdm import tqdm
from sklearn.neighbors import BallTree

# ------------------------------------------------------------
# Compare overlap between network nodes and ALL households
# (NOT just decision_df). We still report overlap with decisions as well.
# ------------------------------------------------------------
def compare_geohash_overlap(decision_df, households_df, network_dict):
    """
    Extract unique geohashes from:
      - households_df (all households, incl. those without any decision)
      - network_dict (all monthly networks)
    And report overlaps with both households and decisions.

    Returns a dict of sets and basic counts.
    """
    # Unique geohashes in the households base table
    if 'geohash8' not in households_df.columns:
        raise ValueError("households_df must contain column 'geohash8'.")
    household_hashes = set(households_df['geohash8'].dropna().astype(str).str.lower().unique())
    print(f"Total unique geohashes in households_df: {len(household_hashes)}")

    # Unique geohashes that actually made a decision (optional diagnostics)
    if 'geohash8' not in decision_df.columns:
        raise ValueError("decision_df must contain column 'geohash8'.")
    decision_hashes = set(decision_df['geohash8'].dropna().astype(str).str.lower().unique())
    print(f"Total unique geohashes in decision_df (deciders only): {len(decision_hashes)}")

    # Collect all unique geohashes from the networks
    network_hashes = set()
    for _, df in network_dict.items():
        g1 = df['group_1'].dropna().astype(str).str.lower()
        g2 = df['group_2'].dropna().astype(str).str.lower()
        network_hashes.update(g1.unique())
        network_hashes.update(g2.unique())
    print(f"Total unique geohashes in network_dict: {len(network_hashes)}")

    # Overlaps
    overlap_with_households = network_hashes & household_hashes
    overlap_with_decisions  = network_hashes & decision_hashes

    print(f"Overlap with households_df: {len(overlap_with_households)} "
          f"({len(overlap_with_households)/len(network_hashes):.2%} of network nodes)")
    print(f"Overlap with decision_df : {len(overlap_with_decisions)} "
          f"({len(overlap_with_decisions)/len(network_hashes):.2%} of network nodes)")

    return {
        'household_hashes': household_hashes,
        'decision_hashes': decision_hashes,
        'network_hashes': network_hashes,
        'overlap_with_households': overlap_with_households,
        'overlap_with_decisions': overlap_with_decisions
    }


# ------------------------------------------------------------
# One-to-one nearest-neighbor mapping:
# Map each network node -> a UNIQUE household in households_df
# (not just decision_df). Ensures injective mapping by distance.
# ------------------------------------------------------------
def fast_match_network_to_households(households_df, network_dict):
    """
    Assign each unique geohash in the network_dict (group_1/group_2)
    to the geographically closest household in households_df,
    ensuring a strict one-to-one mapping (no duplicates on either side).
    Uses BallTree (haversine) for fast NN search.

    Returns:
      aligned_network: same structure as network_dict but nodes replaced by matched households
      mapping_df: DataFrame with columns [network_geohash, household_geohash, distance_m]
    """
    # Basic checks
    if 'geohash8' not in households_df.columns:
        raise ValueError("households_df must contain column 'geohash8'.")

    # Decode helper
    def _decode_series(gh_series):
        def _dec(gh):
            try:
                lat, lon = geohash.decode(str(gh).lower().strip())
                return pd.Series([lat, lon])
            except Exception:
                return pd.Series([np.nan, np.nan])
        return gh_series.apply(_dec)

    # Build the households coordinate table (ensure lat/lon exist)
    hh = households_df[['geohash8']].dropna().copy()
    hh['geohash8'] = hh['geohash8'].astype(str).str.lower().str.strip()
    if not {'lat', 'lon'}.issubset(households_df.columns):
        hh[['lat', 'lon']] = _decode_series(hh['geohash8'])
    else:
        hh = hh.merge(
            households_df[['geohash8', 'lat', 'lon']], on='geohash8', how='left'
        )
    hh = hh.dropna(subset=['lat', 'lon']).drop_duplicates(subset=['geohash8']).reset_index(drop=True)

    # Collect all network nodes
    net_nodes = set()
    for _, df in network_dict.items():
        net_nodes.update(df['group_1'].dropna().astype(str).str.lower().str.strip().unique())
        net_nodes.update(df['group_2'].dropna().astype(str).str.lower().str.strip().unique())
    net_df = pd.DataFrame({'network_geohash': list(net_nodes)})
    net_df[['lat', 'lon']] = _decode_series(net_df['network_geohash'])
    net_df = net_df.dropna(subset=['lat', 'lon']).reset_index(drop=True)

    print(f"Network nodes: {len(net_df)}, Household pool: {len(hh)}")

    # Build BallTree on households
    hh_coords_rad = np.radians(hh[['lat', 'lon']].values)
    tree = BallTree(hh_coords_rad, metric='haversine')

    # Query nearest neighbor for each network node
    net_coords_rad = np.radians(net_df[['lat', 'lon']].values)
    dist_rad, ind = tree.query(net_coords_rad, k=1)
    net_df['nearest_idx'] = ind.flatten()
    net_df['distance_m'] = dist_rad.flatten() * 6371000.0
    net_df['household_geohash'] = hh.iloc[ind.flatten()]['geohash8'].values

    # Enforce one-to-one bijection by greedy distance
    # (keep closest pairs first, drop duplicates on both columns)
    mapping_df = (
        net_df.sort_values('distance_m')
              .drop_duplicates(subset='household_geohash', keep='first')
              .drop_duplicates(subset='network_geohash', keep='first')
              .loc[:, ['network_geohash', 'household_geohash', 'distance_m']]
              .reset_index(drop=True)
    )

    # Apply mapping to all networks
    gh_map = dict(zip(mapping_df['network_geohash'], mapping_df['household_geohash']))
    aligned_network = {}
    for date, df in network_dict.items():
        tmp = df.copy()
        tmp['group_1'] = tmp['group_1'].astype(str).str.lower().str.strip().map(gh_map)
        tmp['group_2'] = tmp['group_2'].astype(str).str.lower().str.strip().map(gh_map)
        tmp = tmp.dropna(subset=['group_1', 'group_2']).reset_index(drop=True)
        aligned_network[date] = tmp

    print(f"Mapped {len(mapping_df)} one-to-one pairs. "
          f"Avg distance: {mapping_df['distance_m'].mean():.1f} m")
    return aligned_network, mapping_df

def match_network_to_decision(decision_df, network_dict):
    """
    For each unique geohash in the network_dict (group_1, group_2),
    find the geographically closest household in decision_df.
    Ensures strict one-to-one mapping (bijective mapping).
    """

    # ---- 1. Decode geohashes to (lat, lon) ----
    def decode(gh):
        try:
            return geohash.decode(gh)
        except Exception:
            return (np.nan, np.nan)

    decision_df = decision_df.copy()
    decision_df[['lat', 'lon']] = decision_df['geohash8'].apply(lambda gh: pd.Series(decode(gh)))

    # collect all network households
    all_network_hashes = set()
    for df in network_dict.values():
        all_network_hashes.update(df['group_1'].dropna().unique())
        all_network_hashes.update(df['group_2'].dropna().unique())
    all_network_hashes = list(all_network_hashes)

    network_coords = pd.DataFrame({
        'network_geohash': all_network_hashes
    })
    network_coords[['lat', 'lon']] = network_coords['network_geohash'].apply(lambda gh: pd.Series(decode(gh)))

    print(f"Total unique network households: {len(network_coords)}")
    print(f"Total decision households: {len(decision_df)}")

    # ---- 2. Compute distance matrix ----
    dist_matrix = np.zeros((len(network_coords), len(decision_df)))
    for i, (lat1, lon1) in tqdm(enumerate(network_coords[['lat', 'lon']].values),
                                total=len(network_coords), desc="Computing distances"):
        for j, (lat2, lon2) in enumerate(decision_df[['lat', 'lon']].values):
            dist_matrix[i, j] = geodesic((lat1, lon1), (lat2, lon2)).meters

    # ---- 3. Greedy one-to-one assignment (minimum distance) ----
    assigned_decisions = set()
    assigned_networks = set()
    mapping = []

    while len(assigned_networks) < len(network_coords) and len(assigned_decisions) < len(decision_df):
        # find global minimum among remaining pairs
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        if np.isinf(dist_matrix[i, j]):
            break

        network_gh = network_coords.iloc[i]['network_geohash']
        decision_gh = decision_df.iloc[j]['geohash8']
        distance = dist_matrix[i, j]

        mapping.append((network_gh, decision_gh, distance))
        assigned_networks.add(i)
        assigned_decisions.add(j)

        # set entire row and column to infinity to block reuse
        dist_matrix[i, :] = np.inf
        dist_matrix[:, j] = np.inf

    mapping_df = pd.DataFrame(mapping, columns=['network_geohash', 'decision_geohash', 'distance_meters'])
    print(f"Matched {len(mapping_df)} one-to-one pairs")

    # ---- 4. Apply mapping back to all network_dict entries ----
    gh_map = dict(zip(mapping_df['network_geohash'], mapping_df['decision_geohash']))

    aligned_network = {}
    for date, df in network_dict.items():
        df = df.copy()
        df['group_1'] = df['group_1'].map(gh_map)
        df['group_2'] = df['group_2'].map(gh_map)
        df = df.dropna(subset=['group_1', 'group_2'])
        aligned_network[date] = df

    print("✅ All network geohashes are now mapped to decision_df households (1-to-1 unique).")
    return aligned_network, mapping_df

import numpy as np
import pandas as pd
import geohash

def compute_geohash_overlap(damage_df, households_df, precision=8):
    """
    Compute the overlap percentage between the geohash codes of damage_df and households_df.

    Parameters
    ----------
    damage_df : pd.DataFrame
        Must contain 'Latitude' and 'Longitude' columns.
    households_df : pd.DataFrame
        Must contain a 'geohash8' column (or geohash column matching the precision).
    precision : int, optional
        Geohash precision level, default = 8.

    Returns
    -------
    dict
        A dictionary with overlap statistics.
    """

    def encode_geohash(lat, lon):
        """Encode a single (lat, lon) pair into geohash."""
        if pd.isna(lat) or pd.isna(lon):
            return np.nan
        try:
            return geohash.encode(float(lat), float(lon), precision=precision).lower()
        except Exception:
            return np.nan

    # --- Step 1. Encode damage_df coordinates into geohash ---
    damage_df = damage_df.copy()
    damage_df['damage_geohash'] = [
        encode_geohash(lat, lon) for lat, lon in zip(damage_df['Latitude'], damage_df['Longitude'])
    ]

    # --- Step 2. Normalize household geohash codes ---
    house_geos = households_df['geohash8'].astype(str).str.lower()

    # --- Step 3. Row-level overlap ---
    valid_mask = damage_df['damage_geohash'].notna()
    match_mask = valid_mask & damage_df['damage_geohash'].isin(house_geos)

    total = len(damage_df)
    valid = int(valid_mask.sum())
    matched = int(match_mask.sum())
    pct = (matched / valid * 100.0) if valid > 0 else 0.0

    # --- Step 4. Unique-geohash-level overlap ---
    damage_unique = set(damage_df.loc[valid_mask, 'damage_geohash'])
    house_unique = set(house_geos)
    inter = len(damage_unique & house_unique)
    pct_unique = (inter / len(damage_unique) * 100.0) if damage_unique else 0.0

    # --- Step 5. Print summary ---
    print("\n--- Geohash Overlap Report ---")
    print(f"Total records in damage_df: {total}")
    print(f"Records with valid coordinates: {valid}")
    print(f"Matched records (same geohash): {matched}")
    print(f"Row-level overlap: {pct:.2f}%")
    print(f"Unique-geohash overlap: {pct_unique:.2f}%  "
          f"(intersection {inter} / unique damage geohash {len(damage_unique)})")

    return damage_df


