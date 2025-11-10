import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def compute_decision_cohesion(decision_df, network_dict, node_to_comm, min_nodes=30):
    """
    Compute Positive and Negative Cohesion for each community based on decision sequences (0/1).
    The filtering rule matches community_analysis(): only communities with node count > min_nodes
    according to node_to_comm are included.
    """
    # --- Step 1: prepare monthly timeline ---
    decision_df = decision_df.copy()
    decision_df['decision_date'] = pd.to_datetime(decision_df['decision_date'])
    months = pd.date_range(decision_df['decision_date'].min(), decision_df['decision_date'].max(), freq='MS')
    month_strs = [d.strftime('%Y-%m') for d in months]

    # --- Step 2: construct binary decision matrix ---
    pivot = pd.DataFrame(0, index=decision_df['geohash8'].unique(), columns=month_strs)
    for _, row in decision_df.iterrows():
        month = row['decision_date'].strftime('%Y-%m')
        if month in pivot.columns:
            pivot.loc[row['geohash8'], month] = 1

    # --- Step 3: build union network to obtain degree weights ---
    edge_list = [df[['group_1', 'group_2']] for df in network_dict.values()]
    all_edges = pd.concat(edge_list, ignore_index=True).drop_duplicates()
    G = nx.from_pandas_edgelist(all_edges, 'group_1', 'group_2')
    degrees = dict(G.degree())

    # --- Step 4: filter large communities (consistent with community_analysis) ---
    comm_map_df = pd.DataFrame(list(node_to_comm.items()), columns=['geohash8', 'community_id'])
    comm_sizes = comm_map_df['community_id'].value_counts()
    large_communities = set(comm_sizes[comm_sizes > min_nodes].index.astype(str))

    # --- Step 5: build node list for each community (filtered) ---
    comm_to_nodes = defaultdict(list)
    for n in pivot.index:
        if n in node_to_comm:
            cid = str(node_to_comm[n])
            if cid in large_communities:
                comm_to_nodes[cid].append(n)

    # --- Step 6: compute decision cohesion ---
    results = {}
    for comm_id, comm_nodes in comm_to_nodes.items():
        if len(comm_nodes) < 2:
            continue

        sub = pivot.loc[comm_nodes]
        corr_matrix = sub.T.corr().fillna(0)
        np.fill_diagonal(corr_matrix.values, 0)

        sub_deg = np.array([degrees.get(n, 0) for n in comm_nodes])
        if sub_deg.sum() == 0:
            continue

        conn_pos = corr_matrix.where(corr_matrix > 0, 0).mean(axis=1)
        conn_neg = corr_matrix.where(corr_matrix < 0, 0).mean(axis=1)

        pos_cohesion = np.average(conn_pos, weights=sub_deg)
        neg_cohesion = np.average(conn_neg, weights=sub_deg)

        results[comm_id] = {
            'Positive_Cohesion': pos_cohesion,
            'Negative_Cohesion': neg_cohesion
        }

    return results


def regression_cohesion_vs_features(correlation_metrix, community_features, community_graph):
    """
    Regress Positive/Negative Cohesion against community-level features.
    Automatically converts correlation_metrix (dict) to DataFrame.
    """
    results = []

    # --- Step 1: convert correlation_metrix dict -> DataFrame ---
    corr_df = pd.DataFrame.from_dict(correlation_metrix, orient='index').reset_index()
    corr_df.rename(columns={'index': 'community_id'}, inplace=True)
    corr_df['community_id'] = corr_df['community_id'].astype(str)

    # --- Step 2: reset index of community_features & community_graph ---
    community_features = community_features.reset_index()
    community_graph = community_graph.reset_index()
    community_features['community_id'] = community_features['community_id'].astype(str)
    community_graph['community_id'] = community_graph['community_id'].astype(str)

    # --- Step 3: merge all on community_id ---
    df = (
        corr_df.merge(community_features, on='community_id', how='inner')
               .merge(community_graph, on='community_id', how='inner')
    )

    # --- Step 4: define variables ---
    dep_vars = ['Positive_Cohesion', 'Negative_Cohesion']
    indep_vars = ['NoDamage', 'Mean_BldgValue', 'Mean_EstLoss', 'avg_degree', 'density', 'avg_clustering']

    # --- Step 5: perform regressions ---
    for dep in dep_vars:
        for var in indep_vars:
            if dep not in df.columns or var not in df.columns:
                continue

            subset = df[[dep, var]].dropna()
            if subset.empty or subset[var].nunique() <= 1:
                continue

            y = subset[dep].astype(float)
            x = sm.add_constant(subset[var].astype(float))

            model = sm.OLS(y, x).fit()

            results.append({
                'Dependent': dep,
                'Independent': var,
                'R2': model.rsquared,
                'p_value': model.pvalues[var]
            })

    return pd.DataFrame(results)


def regression_cohesion_vs_features_plot(correlation_metrix, community_features, community_graph):
    """
    Regress Positive/Negative Cohesion against community-level features.
    Plot fitted lines for Positive_Cohesion only.
    """
    results = []

    # --- Step 1: correlation_metrix -> DataFrame ---
    corr_df = pd.DataFrame.from_dict(correlation_metrix, orient='index').reset_index()
    corr_df.rename(columns={'index': 'community_id'}, inplace=True)
    corr_df['community_id'] = corr_df['community_id'].astype(str)

    # --- Step 2: reset index ---
    community_features = community_features.reset_index()
    community_graph = community_graph.reset_index()
    community_features['community_id'] = community_features['community_id'].astype(str)
    community_graph['community_id'] = community_graph['community_id'].astype(str)

    # --- Step 3: merge ---
    df = (
        corr_df.merge(community_features, on='community_id', how='inner')
               .merge(community_graph, on='community_id', how='inner')
    )

    # --- Step 4: variables ---
    dep_vars = ['Positive_Cohesion', 'Negative_Cohesion']
    indep_vars = ['NoDamage', 'Mean_BldgValue', 'Mean_EstLoss', 'avg_degree', 'density', 'avg_clustering']

    # --- Step 5: regressions + plot Positive_Cohesion ---
    for dep in dep_vars:
        for var in indep_vars:
            if dep not in df.columns or var not in df.columns:
                continue

            subset = df[[dep, var]].dropna()
            if subset.empty or subset[var].nunique() <= 1:
                continue

            y = subset[dep].astype(float)
            x = sm.add_constant(subset[var].astype(float))
            model = sm.OLS(y, x).fit()

            results.append({
                'Dependent': dep,
                'Independent': var,
                'R2': model.rsquared,
                'p_value': model.pvalues[var]
            })

            # --- plot only Positive_Cohesion ---
            if dep == 'Positive_Cohesion':
                plt.figure(figsize=(5, 4))
                sns.regplot(x=var, y=dep, data=subset, ci=None, scatter_kws={'s': 30, 'alpha': 0.6})
                plt.title(f'{dep} vs {var}\nR²={model.rsquared:.3f}, p={model.pvalues[var]:.3e}')
                plt.xlabel(var)
                plt.ylabel(dep)
                plt.tight_layout()
                plt.show()

    return pd.DataFrame(results)