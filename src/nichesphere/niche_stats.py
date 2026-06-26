# %%
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _adjacency_from_graph(g: nx.Graph) -> tuple:
    """Return positive and negative adjacency matrices and the ordered node list.

    The signed edge weights of the co-localization / communication network are
    split into two non-negative matrices so that downstream functions can work
    with each sign independently without repeated sign-checks on every edge.

    Parameters
    ----------
    g : nx.Graph
        Graph with numeric ``weight`` edge attributes (signed).

    Returns
    -------
    A_pos : np.ndarray, shape (n, n)
        Symmetric matrix of positive edge weights (negative weights → 0).
    A_neg : np.ndarray, shape (n, n)
        Symmetric matrix of absolute negative edge weights (positive weights → 0).
    nodes : list
        Ordered list of node labels (index into both matrices).
    """
    nodes = list(g.nodes())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    A_pos = np.zeros((n, n), dtype=float)
    A_neg = np.zeros((n, n), dtype=float)

    for u, v, data in g.edges(data=True):
        w = data.get('weight', 1.0)
        i, j = index[u], index[v]
        if w > 0:
            A_pos[i, j] = w
            A_pos[j, i] = w
        elif w < 0:
            A_neg[i, j] = -w
            A_neg[j, i] = -w

    return A_pos, A_neg, nodes


# ---------------------------------------------------------------------------
# Community edge statistics
# ---------------------------------------------------------------------------

def community_edge_stats(g: nx.Graph, partition: dict) -> dict:
    """Per-community positive/negative edge count and weight statistics.

    For each community the following are computed, distinguishing internal edges
    (both endpoints in the same community) from external ones (endpoints in
    different communities), and positive weights from negative:

    - ``internal_pos_count``, ``internal_pos_weight``
    - ``internal_neg_count``, ``internal_neg_weight``
    - ``external_pos_count``, ``external_pos_weight``
    - ``external_neg_count``, ``external_neg_weight``
    - ``external_by_community`` – nested dict ``{other_community: {pos_count,
      pos_weight, neg_count, neg_weight}}`` for pairwise breakdowns.

    Counts are numbers of edges (unweighted); weights are sums of absolute edge
    weights (``A_neg`` already stores absolute values, per
    :func:`_adjacency_from_graph`).  External edges are attributed to *both*
    communities they connect.

    Parameters
    ----------
    g : nx.Graph
        Signed co-localization network (output of
        :func:`nichesphere.coloc.colocNW`).
    partition : dict
        Mapping of node → community name (e.g. ``vc_map_named``).

    Returns
    -------
    stats : dict
        Keyed by community name; each value is a dict of the metrics listed
        above.
    """
    A_pos, A_neg, nodes = _adjacency_from_graph(g)
    n = len(nodes)
    node_index = {nodes[i]: i for i in range(n)}

    communities = set(partition.values())
    stats = {
        c: {
            'internal_pos_count':  0,
            'internal_pos_weight': 0.0,
            'internal_neg_count':  0,
            'internal_neg_weight': 0.0,
            'external_pos_count':  0,
            'external_pos_weight': 0.0,
            'external_neg_count':  0,
            'external_neg_weight': 0.0,
            'external_by_community': defaultdict(lambda: {
                'pos_count': 0, 'pos_weight': 0.0,
                'neg_count': 0, 'neg_weight': 0.0,
            }),
        }
        for c in communities
    }

    for i_node, j_node in itertools.combinations(nodes, 2):
        i   = node_index[i_node]
        j   = node_index[j_node]
        c_i = partition[i_node]
        c_j = partition[j_node]

        w_pos = A_pos[i, j]
        w_neg = A_neg[i, j]

        if c_i == c_j:
            if w_pos > 0:
                stats[c_i]['internal_pos_count']  += 1
                stats[c_i]['internal_pos_weight'] += w_pos
            if w_neg > 0:
                stats[c_i]['internal_neg_count']  += 1
                stats[c_i]['internal_neg_weight'] += w_neg
        else:
            if w_pos > 0:
                for c in (c_i, c_j):
                    stats[c]['external_pos_count']  += 1
                    stats[c]['external_pos_weight'] += w_pos
                stats[c_i]['external_by_community'][c_j]['pos_count']  += 1
                stats[c_i]['external_by_community'][c_j]['pos_weight'] += w_pos
                stats[c_j]['external_by_community'][c_i]['pos_count']  += 1
                stats[c_j]['external_by_community'][c_i]['pos_weight'] += w_pos
            if w_neg > 0:
                for c in (c_i, c_j):
                    stats[c]['external_neg_count']  += 1
                    stats[c]['external_neg_weight'] += w_neg
                stats[c_i]['external_by_community'][c_j]['neg_count']  += 1
                stats[c_i]['external_by_community'][c_j]['neg_weight'] += w_neg
                stats[c_j]['external_by_community'][c_i]['neg_count']  += 1
                stats[c_j]['external_by_community'][c_i]['neg_weight'] += w_neg

    # convert defaultdicts to plain dicts for cleaner output/printing
    for c in stats:
        stats[c]['external_by_community'] = dict(stats[c]['external_by_community'])

    return stats


def community_edge_stats_df(g: nx.Graph, partition: dict) -> pd.DataFrame:
    """Flat DataFrame version of :func:`community_edge_stats`.

    Returns one row per community with the count and weight columns.  The
    nested ``external_by_community`` breakdown is omitted (call
    :func:`community_edge_stats` directly if you need it).

    Parameters
    ----------
    g : nx.Graph
        Signed co-localization network.
    partition : dict
        Mapping of node → community name.

    Returns
    -------
    df : pd.DataFrame
        Indexed by community name, sorted alphabetically.
    """
    stats = community_edge_stats(g, partition)
    df    = pd.DataFrame.from_dict(stats, orient='index')
    df    = df.drop(columns='external_by_community')
    df.index.name = 'community'
    return df.sort_index()


# ---------------------------------------------------------------------------
# Edge weight distributions
# ---------------------------------------------------------------------------

def community_edge_weight_distributions(g: nx.Graph, partition: dict) -> dict:
    """Raw edge weight lists per community, split by location and sign.

    Uses the same unique-pair iteration as :func:`community_edge_stats` so
    weights correspond one-to-one with the counts computed there.  External
    edges are attributed to *both* communities they connect.

    Parameters
    ----------
    g : nx.Graph
        Signed co-localization network.
    partition : dict
        Mapping of node → community name.

    Returns
    -------
    dist : dict
        Keyed by community name; each value is a dict with four lists:

        - ``'internal_pos'`` – positive weights of internal edges
        - ``'internal_neg'`` – absolute values of negative internal edges
        - ``'external_pos'`` – positive weights of external edges
        - ``'external_neg'`` – absolute values of negative external edges

    Examples
    --------
    >>> dist = nichesphere.niche.community_edge_weight_distributions(gCol, vc_map_named)
    >>> plt.hist(dist['2_FibCore']['internal_pos'])
    """
    A_pos, A_neg, nodes = _adjacency_from_graph(g)
    n          = len(nodes)
    node_index = {nodes[i]: i for i in range(n)}

    communities = set(partition.values())
    dist = {
        c: {'internal_pos': [], 'internal_neg': [],
            'external_pos': [], 'external_neg': []}
        for c in communities
    }

    for i_node, j_node in itertools.combinations(nodes, 2):
        i   = node_index[i_node]
        j   = node_index[j_node]
        c_i = partition[i_node]
        c_j = partition[j_node]

        w_pos = A_pos[i, j]
        w_neg = A_neg[i, j]

        if c_i == c_j:
            if w_pos > 0:
                dist[c_i]['internal_pos'].append(w_pos)
            if w_neg > 0:
                dist[c_i]['internal_neg'].append(w_neg)
        else:
            if w_pos > 0:
                dist[c_i]['external_pos'].append(w_pos)
                dist[c_j]['external_pos'].append(w_pos)
            if w_neg > 0:
                dist[c_i]['external_neg'].append(w_neg)
                dist[c_j]['external_neg'].append(w_neg)

    return dist


def community_signed_weight_distributions(g: nx.Graph, partition: dict) -> dict:
    """Combine positive and negative weights into a single signed distribution.

    Positive weights are kept as-is; negative weights are negated so the
    original score sign is preserved.  This gives one distribution per
    community per edge type (internal / external), suitable for a single pair
    of box plots rather than four separate pos/neg boxes.

    Parameters
    ----------
    g : nx.Graph
        Signed co-localization network.
    partition : dict
        Mapping of node → community name.

    Returns
    -------
    signed : dict
        Keyed by community name; each value is a dict with keys
        ``'internal'`` and ``'external'``, each a list of signed weight values.
    """
    dist   = community_edge_weight_distributions(g, partition)
    signed = {}
    for c, d in dist.items():
        signed[c] = {
            'internal': d['internal_pos'] + [-w for w in d['internal_neg']],
            'external': d['external_pos'] + [-w for w in d['external_neg']],
        }
    return signed


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------

def pairwise_ranksums_int_ext(data_int: list, data_ext: list, communities: list,
                               MTcorrection: str = 'fdr_bh') -> pd.DataFrame:
    """Wilcoxon rank-sum test comparing internal vs external edge weight
    distributions for each community, with multiple-testing correction.

    For each community the test asks whether the internal edge weight
    distribution differs significantly from the external one — a proxy for
    whether the niche is more cohesively connected internally than its
    cross-niche connections would suggest.

    Communities with fewer than 2 edges in either distribution are skipped.

    Parameters
    ----------
    data_int : list of lists
        Internal signed edge weight distributions, one list per community
        (same order as ``communities``).
    data_ext : list of lists
        External signed edge weight distributions, one list per community.
    communities : list of str
        Community/niche names, same order as ``data_int`` / ``data_ext``.
    MTcorrection : str, default ``'fdr_bh'``
        Multiple-testing correction method passed to
        ``statsmodels.multipletests`` (e.g. ``'fdr_bh'``, ``'holm'``,
        ``'bonferroni'``).

    Returns
    -------
    df : pd.DataFrame
        One row per tested community with columns ``niche``, ``statistic``,
        ``p_value``, and ``p_value_corrected``.
    """
    usable_int = {c: d for c, d in zip(communities, data_int) if len(d) >= 2}
    usable_ext = {c: d for c, d in zip(communities, data_ext) if len(d) >= 2}
    usable     = sorted(set(usable_int) & set(usable_ext))

    rows = []
    for c in usable:
        stat, p = ranksums(usable_int[c], usable_ext[c])
        rows.append({'niche': c, 'statistic': stat, 'p_value': p})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df['p_value_corrected'] = multipletests(
        df['p_value'].values, alpha=0.05, method=MTcorrection,
        maxiter=1, is_sorted=False, returnsorted=False
    )[1]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_internal_external_boxplots(g: nx.Graph, partition: dict, ax=None):
    """Box plots of signed edge weights (internal vs external) per community.

    For each detected niche/community, two side-by-side box plots show the
    distribution of signed edge weights for edges connecting nodes *within*
    the community (internal, blue) and edges connecting nodes *across*
    communities (external, orange).  Positive weights represent enriched
    co-localization; the plot therefore visualises how cohesive each niche is
    relative to its cross-niche connections.

    Parameters
    ----------
    g : nx.Graph
        Signed co-localization network (output of
        :func:`nichesphere.coloc.colocNW`).
    partition : dict
        Mapping of node → community name (e.g. ``vc_map_named``).
    ax : matplotlib.axes.Axes or None, default None
        Axes to draw on; a new figure is created when ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the box plots.
    """
    signed      = community_signed_weight_distributions(g, partition)
    communities = sorted(signed.keys())

    internal_data = [signed[c]['internal'] for c in communities]
    external_data = [signed[c]['external'] for c in communities]

    n                  = len(communities)
    positions_internal = np.arange(n) * 2.0
    positions_external = positions_internal + 0.8

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, n * 1.5), 5))

    bp_internal = ax.boxplot(
        internal_data, positions=positions_internal, widths=0.6,
        patch_artist=True, tick_labels=None,
    )
    bp_external = ax.boxplot(
        external_data, positions=positions_external, widths=0.6,
        patch_artist=True, tick_labels=None,
    )

    for patch in bp_internal['boxes']:
        patch.set_facecolor('#4C72B0')
        patch.set_alpha(0.7)
    for patch in bp_external['boxes']:
        patch.set_facecolor('#DD8452')
        patch.set_alpha(0.7)

    ax.set_xticks(positions_internal + 0.4)
    ax.set_xticklabels(communities)
    ax.set_xlabel('niche')
    ax.set_ylabel('edge weight')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.legend(
        [bp_internal['boxes'][0], bp_external['boxes'][0]],
        ['internal', 'external'],
        loc='best',
    )
    ax.set_title('Internal vs external edge weight distributions by niche')
    return ax
