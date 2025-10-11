import pandas as pd
from itertools import combinations


def build_group_network(graph_df, result_df_subset, start_date):
    # --------------------------
    # 1. Read the CSV file and generate the individual_links list
    # --------------------------
    # Keep only user pairs predicted as connected, and extract user_u, user_v, and connection type columns
    individual_links = graph_df[['user_u', 'user_v', 'connection type']].values.tolist()
    print("Total individual links (predicted connected):", len(individual_links))
    # print(individual_links)
    # --------------------------
    # 2. Build a mapping from device to group (group geohash)
    # Assume result_df_subset contains columns device_id and group_geohash_8
    # --------------------------
    device_to_group = result_df_subset.set_index('device_id')['group_geohash_8'].to_dict()

    # --------------------------
    # 3. Count the number of devices in each group
    # --------------------------
    group_device_count = result_df_subset['group_geohash_8'].value_counts().to_dict()

    # --------------------------
    # 4. Traverse individual_links to generate inter-group connection counts
    #    Only consider connections between different groups.
    #    For each group pair, if there is at least one bonding link (conn_type==1), mark type as 1.
    # --------------------------
    group_link_counts = {}
    for d1, d2, conn_type in individual_links:
        if d1 in device_to_group and d2 in device_to_group:
            h1 = device_to_group[d1]
            h2 = device_to_group[d2]
            if h1 != h2:
                # Ensure consistent order
                group_pair = tuple(sorted([h1, h2]))
                if group_pair not in group_link_counts:
                    group_link_counts[group_pair] = {'count': 0, 'type': 0}
                group_link_counts[group_pair]['count'] += 1
                # If at least one bonding link exists, set type to 1
                if conn_type == 1:
                    group_link_counts[group_pair]['type'] = 1

    # --------------------------
    # 5. Generate list of group candidate pairs
    # --------------------------
    group_candidate_pair = list(group_link_counts.keys())
    print("Number of group candidate pairs:", len(group_candidate_pair))

    # --------------------------
    # 6. Construct group_edges DataFrame and compute avg_link (average connection strength)
    #    avg_link = link_count / (number of devices in group 1 + number of devices in group 2)
    # --------------------------
    group_edges_data = []
    for (h1, h2), info in group_link_counts.items():
        link_count = info['count']
        group_type = info['type']
        # Compute the number of possible device pairs (sum of device counts from both groups)
        num_possible = group_device_count.get(h1, 0) * group_device_count.get(h2, 0)
        avg_link = link_count / num_possible if num_possible > 0 else 0
        group_edges_data.append({
            "group_1": h1,
            "group_2": h2,
            "link_count": link_count,
            "num_possible": num_possible,
            "avg_link": avg_link,
            "type": group_type
        })

    group_edges = pd.DataFrame(group_edges_data)

    # --------------------------
    # Output results
    # --------------------------
    # print('How many devices are there in each group:')
    # print(group_device_count)
    # print("group candidate pairs:")
    # print(group_candidate_pair)


    group_edges['group_1_number'] = group_edges['group_1'].map(group_device_count)
    group_edges['group_2_number'] = group_edges['group_2'].map(group_device_count)
    # Identify all groups and those already in edges
    all_groups = set(result_df_subset['group_geohash_8'])
    groups_in_edges = set(group_edges['group_1']) | set(group_edges['group_2'])
    print('All Group',len(all_groups))
    print("Unique Device Id count:", len(result_df_subset))
    print('Group that have links:',len(groups_in_edges))
    # Find isolated groups (no edges)
    isolated_groups = all_groups - groups_in_edges

    # Create rows for isolated groups
    isolated_rows = []
    for h in isolated_groups:
        isolated_rows.append({
            'group_1': h,
            'group_2': None,
            'link_count': None,
            'num_possible': None,
            'avg_link': None,
            'type': None,
            'group_1_number': group_device_count.get(h, None),
            'group_2_number': None
        })

    # Append to group_edges
    isolated_df = pd.DataFrame(isolated_rows)
    group_edges = pd.concat([group_edges, isolated_df], ignore_index=True)

    group_edges.to_csv(f"results/Group_social_network_{start_date.date()}.csv", index=False)
    return group_edges