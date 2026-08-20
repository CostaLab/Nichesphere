# %%
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import sklearn
import scanpy as sc
from matplotlib.colors import ListedColormap

from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import NearestNeighbors


def cellCatContained(pair, cellCat):
    """Check if a cell group (niche/type/category) is contained in a cell type pair

    Parameters
    ----------
    pair : list
        cell type list (usually cell type pairs in the form [cellTypeA,cellTypeB])

    cellCat : list
        list of cell types in a cell group (niche/type/category)

    Returns
    -------
    True or False
    """
    
    contained=[cellType in pair for cellType in cellCat]
    return True in contained

# %%

def getColocProbs(CTprobs: pd.DataFrame, spotSamples: pd.Series, spotweights: pd.Series = None) -> pd.DataFrame:
    """Vectorized calculation of cell type pair co-localization probabilities per sample
    (sum across spots of probabilities of each cell type pair being in the same spot)
    Parameters
    ----------
    CTprobs : pd.DataFrame
        Dataframe of cell type probabilities per spot
    spotSamples : pd.Series
        Series indicating the sample to which each spot belongs, with spot ids as index.
    spotweights : pd.Series (default: None)
        Series of weights per spot with spot ids as index.

    Returns
    -------
    CTcolocalizationP : pd.DataFrame
        concatenated dataframes of cell type pairs co-localization probabilities per sample
    """
    results = []
    for smple, group in CTprobs.groupby(spotSamples):
        n_spots = len(group)
        if n_spots == 0:
            continue

        if spotweights is None:
            # Matrix multiplication: (CellTypes x Spots) @ (Spots x CellTypes) / n_spots
            coloc_mat = (group.T.values @ group.values) / n_spots
        else:
            # align weights to this group's spots, same order
            w = spotweights.loc[group.index].values
            w_sum = w.sum()
            # weight the rows once, then matmul: sum_i w_i * ct_i * col_i
            coloc_mat = (group.T.values @ (group.values * w[:, None])) / w_sum

        df_smple = pd.DataFrame(coloc_mat, index=group.columns, columns=group.columns)
        df_smple["sample"] = smple
        results.append(df_smple)

    return pd.concat(results, axis=0) if results else pd.DataFrame()

# %%

def compute_sc_spatial_knn_coloc_matrix(adata, cluster_col, k=5):

    """
    Compute a kNN-based co-localization probability matrix between cell type pairs.

    For each cell, the k nearest spatial neighbors are identified. The co-localization
    matrix counts how often each cluster appears in the neighborhood of each other
    cluster, then normalizes counts to obtain interaction probabilities.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial AnnData object. Must contain cell centroid coordinates in
        ``adata.obsm['spatial']`` and a categorical cluster annotation column in
        ``adata.obs``.
    cluster_col : str
        Name of the categorical column in ``adata.obs`` containing cluster labels.
    k : int, default 5
        Number of nearest spatial neighbors to consider per cell (excluding self).

    Returns
    -------
    pd.DataFrame
        Symmetric co-localization probability matrix of shape
        (n_clusters, n_clusters). Entry (i, j) is the probability of finding
        cluster j in the k-nearest neighborhood of cluster i, normalized so
        that all entries sum to 1.

    Notes
    -----
    The ``cluster_col`` column must be of dtype ``category``; cluster order
    follows ``adata.obs[cluster_col].cat.categories``.

    Examples
    --------
    >>> coloc = compute_sc_spatial_knn_coloc_matrix(adata, cluster_col='cell_type', k=10)
    >>> sns.heatmap(coloc, cmap='Blues')
    """
    # Extract centroids of polygons
    coords=adata.obsm['spatial']

    # Fit kNN model
    ### Try radius??
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # Get cluster labels
    cluster_labels = adata.obs[cluster_col].values
    #unique_clusters = np.unique(cluster_labels)
    unique_clusters = adata.obs[cluster_col].cat.categories

    # Initialize colocalization matrix
    cluster_index = {c: i for i, c in enumerate(unique_clusters)}
    colocalization_matrix = np.zeros((len(unique_clusters), len(unique_clusters)), dtype=int)

    # Populate the matrix
    for i, neighbors in enumerate(indices):
        cluster_i = cluster_labels[i]
        for neighbor in neighbors[1:]:  # Exclude self (first neighbor)
            cluster_j = cluster_labels[neighbor]
            colocalization_matrix[cluster_index[cluster_i], cluster_index[cluster_j]] += 1
    W = colocalization_matrix
    W = W/W.sum() ## Prob to find an boundary
    # Convert to a Pandas DataFrame for readability
    colocalization_df = pd.DataFrame(W, index=unique_clusters, columns=unique_clusters)
    
    return colocalization_df

# %%

def compute_sc_spatial_radius_coloc_matrix(adata, cluster_col, radius=40.0):

    """
    Compute a radius-based co-localization probability matrix between cell type pairs.

    For each cell, all neighbors within a fixed spatial radius are identified.
    Interaction counts are symmetrized (so that A→B and B→A are treated equally)
    and normalized to obtain interaction probabilities.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial AnnData object. Must contain cell centroid coordinates in
        ``adata.obsm['spatial']`` and a categorical cluster annotation column in
        ``adata.obs``.
    cluster_col : str
        Name of the categorical column in ``adata.obs`` containing cluster labels.
    radius : float, default 40.0
        Spatial radius within which two cells are considered co-localized.
        Units match those of ``adata.obsm['spatial']`` (e.g. pixels or micrometers).

    Returns
    -------
    pd.DataFrame
        Symmetric co-localization probability matrix of shape
        (n_clusters, n_clusters). Entry (i, j) is the symmetrized probability
        of finding cluster j within ``radius`` of cluster i, normalized so
        that all entries sum to 1.

    Notes
    -----
    The ``cluster_col`` column must be of dtype ``category``; cluster order
    follows ``adata.obs[cluster_col].cat.categories``.

    Unlike ``compute_sc_spatial_knn_coloc_matrix``, this function uses a fixed spatial
    radius rather than a fixed number of neighbors, making it more suitable for
    datasets where cell density varies across the tissue.

    Examples
    --------
    >>> coloc = compute_sc_spatial_radius_coloc_matrix(adata, cluster_col='cell_type', radius=300.0)
    >>> sns.heatmap(coloc, cmap='Blues')
    """

    coords = adata.obsm['spatial']
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(coords)
    indices = nbrs.radius_neighbors(coords, return_distance=False)

    cluster_labels = adata.obs[cluster_col].values
    categories = adata.obs[cluster_col].cat.categories
    cluster_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    n = len(categories)
    mat = np.zeros((n, n))

    for i, neighbors in enumerate(indices):
        # map current cell label to index
        row = cluster_to_idx[cluster_labels[i]]
        for nb in neighbors:
            if i == nb: continue # ignore self
            col = cluster_to_idx[cluster_labels[nb]]
            mat[row, col] += 1

    # Symmetrize and Normalize
    mat = (mat + mat.T) / 2
    total = mat.sum()
    if total > 0:
        mat = mat / total

    return pd.DataFrame(mat, index=categories, columns=categories)


# %%
def reshapeColoc(CTcoloc, oneCTinteractions='', complete=1):   
    """Transforms matrix obtained with getColocProbs into a matrix of CT pairs x samples
    CTcoloc=previously obtained colocalisation matrix from getColocprobs
    complete=list with repeated values (ct1_x_ct2 and ct2_x_ct1)
    
    Parameters
    ----------
    CTcoloc : pd.DataFrame
        Concatenated dataframes of cell type pairs co-localization probabilities per sample (obtained via getColocProbs())
    oneCTinteractions : list
        list of single cell interactions (celltype-celltype)
    complete : flag (default: 1)
        indicates if the co-localization matrices are complete (1) (not just half) , with repeated values at ct1_x_ct2 and ct2_x_ct1
        or half matrices are used as input (0)
    
    Returns
    -------
    colocPerSample1 : pd.DataFrame
        Probabilities of each cell type pair per sample
    """

    # 1. Identify cell type columns while excluding 'sample'
    ct_cols = [c for c in CTcoloc.columns if c != "sample"]
    ct_rows = CTcoloc.index.unique().tolist()

    # 2. Extract cell-type 1 from index and prepare long format
    df = CTcoloc.reset_index().rename(columns={"index": "ct1"})

    df_long = df.melt(
        id_vars=["sample", "ct1"],
        value_vars=ct_cols,
        var_name="ct2",
        value_name="prob",
    )

    # 3. Create 'ct1-ct2' interaction pair string
    df_long["pair"] = df_long["ct1"] + "-" + df_long["ct2"]

    # 4. Generate EXACT column order as original nested loops
    ordered_pairs = []
    if complete == 0:
        # Original loop: for ct in columns ... for ct2 in columns[i:]
        for i, ct in enumerate(ct_cols):
            for ct2 in ct_cols[i:]:
                ordered_pairs.append(f"{ct}-{ct2}")
        # Filter df_long to only include these upper triangle pairs
        df_long = df_long[df_long["pair"].isin(set(ordered_pairs))]
    else:
        # Original loop: for ct in columns ... for ct2 in columns
        for ct in ct_cols:
            for ct2 in ct_cols:
                ordered_pairs.append(f"{ct}-{ct2}")

    # 5. Pivot and reindex columns to guarantee original exact column ordering
    colocPerSample1 = df_long.pivot(
        index="sample", columns="pair", values="prob"
    )

    # Re-order columns to match original nested loop sequence
    existing_pairs = [p for p in ordered_pairs if p in colocPerSample1.columns]
    colocPerSample1 = colocPerSample1[existing_pairs]

    # Clean up index metadata
    colocPerSample1.index.name = None
    colocPerSample1.columns.name = None

    # 6. Apply *2 multiplier for complete == 0 on non-single CT interactions
    if complete == 0 and oneCTinteractions:
        diff_cols = np.setdiff1d(colocPerSample1.columns, oneCTinteractions)
        colocPerSample1[diff_cols] = colocPerSample1[diff_cols] * 2

    return colocPerSample1

# %%
def diffColoc_test(coloc_pair_sample, sampleTypes, exp_condition, ctrl_condition):
    """ Differential co-localization test with table of scores and p-values as output

    Parameters
    ----------
    coloc_pair_sample : pd.DataFrame
        coloc per cell type pair per sample table
    exp_condition : string
        non control phenotype to test
    ctrl_condition : string
        control phenotype
    sampleTypes : pd.DataFrame
        dataframe with sample names and sample types columns named "sample" and "sampleType"

    Returns
    -------
    df : pd.DataFrame
        Dataframe of ranksums test statistic and p-value per cell type pair
    """
    pvals=[scipy.stats.ranksums(coloc_pair_sample.loc[coloc_pair_sample.index[sampleTypes.sampleType==exp_condition],c], 
                                        coloc_pair_sample.loc[coloc_pair_sample.index[sampleTypes.sampleType==ctrl_condition],c]).pvalue for c in coloc_pair_sample.columns]
    stat=[scipy.stats.ranksums(coloc_pair_sample.loc[coloc_pair_sample.index[sampleTypes.sampleType==exp_condition],c], 
                                        coloc_pair_sample.loc[coloc_pair_sample.index[sampleTypes.sampleType==ctrl_condition],c]).statistic for c in coloc_pair_sample.columns]


    df=pd.DataFrame([coloc_pair_sample.columns, stat, pvals], index=['pairs', 'statistic', 'p-value']).T
    df.index=df.pairs
    return df

#%%
def spatialNichePlot(adata, niche_dict, CTprobs, spot_size=1, legend_fontsize=7, title="", legend_loc='right margin', save_name='test.pdf', niche_colors=None, ax=None, vmin=0, vmax=None):
    """ Plot niches in spatial data (MERFISH / visium slices)

    Parameters
    ----------
    adata : AnnData
        sample specific spatial anndata object
    niche_dict : dict
        dictionary with niche names as keys and lists of their corresponding cell types as values 
    CTprobs : pd.DataFrame
        sample specific cell type probabilities per spot
    spot_size : int (default: 1)
        size of spots in the spatial plot
    legend_fontsize : int (default: 7)
    title : str
    legend_loc : str (default: 'right margin')
         legend location , can switch to  'on data' 
    save_name : str (default: 'test.pdf')
    niche_colors : pd.Series
        series of colors with niche names as indexes
    ax : matplotlib.axes.Axes (default: None)
        The axes object to draw the plot onto.
    vmin : int (default: 0)
        Minimum value in the color scale
    vmax : int (default: None)
        Maximum value in the color scale

    Returns
    -------
    Spatial plot where spots are colored by cell type with highest proportion
    """
    tmp=adata.copy()
    for niche in list(niche_dict.keys()):
        tmp.obs[niche]=CTprobs[niche_dict[niche]].sum(axis=1)
    niche_props=tmp.obs[list(niche_dict.keys())]
    tmp.obs['max_niche']= [niche_props.columns[np.argmax(niche_props.loc[idx])] for idx in niche_props.index]

    tmp.obs.max_niche=tmp.obs.max_niche.astype('category')
    for c in np.setdiff1d(list(niche_dict.keys()),tmp.obs.max_niche.cat.categories):
        tmp.obs.max_niche = tmp.obs.max_niche.cat.add_categories(c)
    tmp.obs.max_niche=tmp.obs.max_niche.cat.reorder_categories(list(niche_dict.keys()))

    
    if niche_colors is not None:
        tmp.uns['max_niche_colors']=niche_colors
        
    sc.pl.spatial(tmp, color='max_niche', img_key=None, library_id=None, spot_size=spot_size, legend_fontsize=legend_fontsize, title = title,legend_loc=legend_loc, 
                  save=save_name, ax=ax)

#%%

def spatialCTPlot(adata, cell_types, CTprobs=None, maxCT_col=None, spot_size=1, legend_fontsize=7, title="", legend_loc='right margin', save_name='test.pdf', ax=None):
    """ Plot cell types in spatial data (MERFISH / visium slices)

    Parameters
    ----------
    adata : AnnData
        sample specific spatial anndata object
    cell_types : pd.Series
        categorical series of cell types
    CTprobs : pd.DataFrame (default: None)
        sample specific cell type probabilities per spot (not needed if there is a cell type column (maxCT_col) in the anndata obs)
    maxCT_col : string (default: None)
        cell type column in anndata object
    spot_size : int (default: 1)
        size of spots in the spatial plot
    legend_fontsize : int (default: 7)
    title : str
    legend_loc : str (default: 'right margin')
         legend location , can switch to  'on data' 
    save_name : str (default: 'test.pdf')
    ax : matplotlib.axes.Axes (default: None)
        The axes object to draw the plot onto.

    Returns
    -------
    Spatial plot where spots are colored by cell type with highest proportion
    """
    tmp=adata.copy()

    if maxCT_col is None:
        tmp.obs['maxCT']=[CTprobs.columns[np.argmax(CTprobs.loc[idx])] for idx in CTprobs.index]
        tmp.obs.maxCT=tmp.obs.maxCT.astype('category')
        
        for c in np.setdiff1d(cell_types.cat.categories,tmp.obs.maxCT.cat.categories):
            tmp.obs.maxCT = tmp.obs.maxCT.cat.add_categories(c)
        tmp.obs.maxCT=tmp.obs.maxCT.cat.reorder_categories(cell_types.cat.categories)
    else:
        tmp.obs['maxCT']=tmp.obs[maxCT_col]
    
    sc.pl.spatial(tmp, color='maxCT', img_key=None, library_id=None, spot_size=spot_size, legend_fontsize=legend_fontsize, title = title,legend_loc=legend_loc, 
                  save=save_name, ax=ax)
    
#%%
def spatialSingleCTPlot(adata, cell_type, CTprobs, spot_size=1, legend_fontsize=7, title="", legend_loc='right margin', save_name='test.pdf', ax=None, vmin=0, vmax=None, cmap='magma'):
    """ Plot a single cell type proportions in spatial data (MERFISH / visium slices)
    
    Parameters
    ----------
    adata : AnnData
        sample specific spatial anndata object
    CTprobs : pd.DataFrame (default: None)
        sample specific cell type probabilities per spot (not needed if there is a cell type column (maxCT_col) in the anndata obs)
    cell_type : str 
        cell type to plot
    spot_size : int (default: 1)
        size of spots in the spatial plot
    legend_fontsize : int (default: 7)
    title : str
    legend_loc : str (default: 'right margin')
         legend location , can switch to  'on data' 
    save_name : str (default: 'test.pdf')
    ax : matplotlib.axes.Axes (default: None)
        The axes object to draw the plot onto.
    vmin : int (default: 0)
        Minimum value in the color scale
    vmax : int (default: None)
        Maximum value in the color scale
    cmap : str (default: 'magma')
        Name of the color map to be used

    Returns
    -------
    Spatial plot where spots are colored by probability of the selected cell type
    """
    tmp=adata.copy()
    
    tmp.obs['cell_to_plot']=CTprobs[cell_type]
    sc.pl.spatial(tmp, color='cell_to_plot', img_key=None, library_id=None, spot_size=spot_size, legend_fontsize=legend_fontsize, title = title,legend_loc=legend_loc, 
                  save=save_name, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)


#%%

def OvsE_coloc_test(observedColocProbs, expectedColocProbs, cell_types, testDistribution, oneCTinteractions, p=0.05):
    """ Observed vs Expected log2 ratios filtered by p-value obtained from comparing them against a background distribution
    
    Parameters
    ----------
    observedColocProbs : pd.Series
        observed cell type pair co-localization probabilities in a sample/condition
    expectedColocProbs : pd.Series
        expected cell type pair co-localization probabilities in a sample/condition
    cell_types : list 
        list of cell types (sorted as in observedColocProbs)
    testDistribution : list
        distribution of log2 observed vs expected ratios obtained from random sampling pairs of single cells
        from scRNA data from the same sample/condition (obtained generally from function 'PIC_OEratios_BGdist')
        from the tl module
    oneCTinteractions : list
        list of single cell interactions (celltype-celltype)
    p : float (default: 0.05)
        values must have a p-value lower than this to be considered significant
    Returns
    -------
    OvsE_HMdf : pd.DataFrame
        cell types x cell types data frame of log2 observed/expected significant values (scores) for each cell cell interaction
    """
    
    OvsE=observedColocProbs/expectedColocProbs
    OvsE_HM=np.log2(OvsE)
    
    OvsE_HM[(OvsE_HM>np.quantile(np.log2(testDistribution), p/2)) & (OvsE_HM<np.quantile(np.log2(testDistribution), 1-(p/2)))]=0
    OvsE_HM[oneCTinteractions]=0
    
    OvsE_HMdf=pd.DataFrame(np.array(OvsE_HM).reshape(-1, len(cell_types)))
    OvsE_HMdf.columns=cell_types
    OvsE_HMdf.index=cell_types
    return OvsE_HMdf

#%%

def OvsE_coloc_test_adjPval(observedColocProbs, expectedColocProbs, testDistribution_df, cell_types,
                    oneCTinteractions, p=0.05, MTcorrection='fdr_bh', pseudocount=1e-10):
    """Observed vs Expected log2 ratios filtered by adjusted p-value from a per-pair background distribution.

    Parameters
    ----------
    observedColocProbs : pd.Series
        Observed cell type pair co-localization probabilities in a sample/condition.
    expectedColocProbs : pd.Series
        Expected cell type pair co-localization probabilities in the same sample/condition.
    testDistribution_df : pd.DataFrame
        Background distribution of O/E ratios: rows are bootstrap iterations, columns are cell type
        pairs (as returned by ``nichesphere.tl.get_PIC_BG_OEratios_DF``).
    cell_types : list
        List of cell types in the same order as ``observedColocProbs``.
    oneCTinteractions : list
        List of same-cell-type interactions (``cellType-cellType``) to exclude from scoring.
    p : float, default 0.05
        Adjusted p-value threshold; pairs above this are set to 0.
    MTcorrection : str, default 'fdr_bh'
        Multiple-testing correction method passed to ``statsmodels.multipletests``.
    pseudocount : float, default 1e-10
        Small value added to numerator and denominator before computing O/E ratios to avoid
        division by zero or log(0).

    Returns
    -------
    OvsE_HMdf : pd.DataFrame
        Cell types × cell types matrix of log2(O/E) scores; non-significant and
        same-cell-type values are set to 0.
    statsDF : pd.DataFrame
        Per-pair statistics with columns ``statistic`` (log2 O/E), ``pvalue``, and
        ``adjusted_pvalue``.
    """
    OvsE = (observedColocProbs + pseudocount) / (expectedColocProbs + pseudocount)
    OvsE_HM = np.log2(OvsE)

    pvals = pd.Series([
        scipy.stats.percentileofscore(np.log2(testDistribution_df[x] + pseudocount),
                                      OvsE_HM[x], kind='weak') / 100
        for x in OvsE_HM.index
    ])
    pvals[pvals > 0.5] = 1 - pvals[pvals > 0.5]
    pvals = pvals * 2
    pvals_corr = multipletests(pvals, alpha=0.05, method=MTcorrection,
                               maxiter=1, is_sorted=False, returnsorted=False)[1]

    OvsE_HM[pvals_corr > p] = 0
    OvsE_HM[oneCTinteractions] = 0

    OvsE_HMdf = pd.DataFrame(np.array(OvsE_HM).reshape(-1, len(cell_types)))
    OvsE_HMdf.columns = cell_types
    OvsE_HMdf.index = cell_types

    statsDF = pd.DataFrame({
        'statistic': list(OvsE_HM),
        'pvalue': list(pvals),
        'adjusted_pvalue': list(pvals_corr)
    })
    statsDF.index = observedColocProbs.index

    return OvsE_HMdf, statsDF
#%%

def colocNW_old(x_diff,adj, cell_group, group=None, group_cmap='tab20', ncols=20, clist=None, 
            nodeSize=None, legend_ax=[0.7, 0.05, 0.15, 0.2], layout='neato', lab_spacing=9, thr=0, alpha=1, fsize=(8,8), pos=None, 
            edge_scale=1):
    """ (Differential) co-localization network

    Parameters
    ----------
    xdiff : pd.DataFrame
        cell types x cell types data frame of significant 
        scores for each cell cell interaction
    adj : pd.DataFrame
        cell types x cell types adjacency matrix (calculated from the cell cell 
        interaction scores)
    cell_group : dict
        dictionary with niche names as keys and lists of their corresponding cell types as values 
    group : list (default: None)
        list of nodes whose interaction will be highlighted
    group_cmap : str (default: 'tab20')
        name of the color map from which the niche colors will be taken
    ncols : int (default: 20)
        number of colors for the group_cmap
    clist : list (default: None)
        alternatively , one can input a list of niche colors
    nodeSize : str (default: None)
        value that will define the size of the nodes. Options are 'betweeness', 
        'pagerank' (network statistics)
    legend_ax : list (default: [0.7, 0.05, 0.15, 0.2])
        legend position in the form [x0, y0, width, height]
    layout : str (default: 'neato')
        name of the layout to be used. Options are 'neato', 'dot', 'kamada_kawai', 
        'spring', 'spectral', 'circular', 'force_atlas2', 'fruchterman_reingold' and 'random')
    lab_spacing : int (default: 9)
        spacing between labels and nodes
    thr : float (default: 0)
        edge weights absolute value must be above this value for the edge to be shown
    alpha : float (default: 1)
        edge transparency (from 0 to 1)
    fsize : tuple
        figure size in the form (x,y)
    pos : dict
        dictionary containing the calculated 2D positions (x, y coordinates) for every node
    edge_scale : float
        factor to scale the edge width

    Returns
    -------
    gCol : nx.Graph
        Graph object with cell cell interaction scores as weights
    Network plot
    """
    
    ## Make color maps
    cmap = plt.cm.RdBu
    cmap3 = cmap(np.arange(cmap.N))
    cmap3[:,-1] = np.linspace(0, alpha, cmap.N)
    c1=cmap3.copy()
    cmap3 = ListedColormap(cmap3)
    
    cmap = plt.cm.RdBu_r
    cmap4 = cmap(np.arange(cmap.N))
    cmap4[:,-1] = np.linspace(0, alpha, cmap.N)
    c2=cmap4.copy()
    cmap4 = ListedColormap(cmap4)

    colors = np.vstack((np.flip(c1[128:256], axis=0), c2[128:256]))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    cmap=mcolors.LinearSegmentedColormap.from_list("WhiteGray",['white','lightgrey'])
    graycmp = cmap(np.arange(cmap.N))
    graycmp[:,-1] = np.linspace(0, alpha-0.2, cmap.N)
    c3=graycmp.copy()
    graycmp = ListedColormap(graycmp)
    
    #cell groups cmap
    cmap = plt.colormaps[group_cmap].resampled(ncols)
    if clist == None:
        cgroup_cmap=[mcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    else:
        cgroup_cmap=clist
    
    gCol=nx.from_pandas_adjacency(adj, create_using=nx.Graph)

    ## Edge thickness (NEW)
    for x in list(gCol.edges):
        gCol[x[0]][x[1]]['weight'] = np.abs(x_diff.loc[x[0], x[1]])
    
    ## Node color groups
    color_group=pd.Series(list(gCol.nodes))
    i=0
    for k in list(cell_group.keys()):
        color_group[[cellCatContained(pair=p, cellCat=cell_group[k]) for p in color_group]]=cgroup_cmap[i]
        i=i+1

    ### different layouts
    if layout=='neato':
        pos = nx.drawing.nx_agraph.graphviz_layout(gCol,prog='neato')
    if layout=='dot':
        pos = nx.drawing.nx_agraph.graphviz_layout(gCol,prog='dot')
    if layout=='kamada_kawai':
        pos = nx.drawing.kamada_kawai_layout(gCol)
    if layout=='spring':
        pos = nx.drawing.spring_layout(gCol)
    if layout=='spectral':
        pos = nx.drawing.spectral_layout(gCol)
    if layout=='circular':
        pos = nx.drawing.circular_layout(gCol)
    if layout=='force_atlas2':
        pos = nx.drawing.forceatlas2_layout(gCol)
    if layout=='fruchterman_reingold':
        pos = nx.drawing.fruchterman_reingold_layout(gCol)
    if layout=='random':
        pos = nx.drawing.random_layout(gCol)

    if pos!=None:
        pos=pos

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1]+lab_spacing)
    
    to_remove=[(a,b) for a, b, attrs in gCol.edges(data=True) if np.abs(attrs["weight"]) <=thr]
    gCol.remove_edges_from(to_remove)

    ## Edge colors based on diff coloc
    edgeCols=pd.Series(['lightblue' if x_diff.loc[x[0], x[1]]<0 else 'orange' for x in list(gCol.edges)])
    edgeCols.index=[x[0]+'->'+x[1] for x in list(gCol.edges)]
    
    orange_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'orange']
    blue_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'lightblue']

    #normalised scores
    weights = nx.get_edge_attributes(gCol,'weight').values()
    inter=pd.Series(np.abs(pd.Series(list(weights))))
    inter.index=edgeCols.index

    #classify edges by color
    if group!=None:
        edgeCols[[cellCatContained(pair=[x.split('->')[0], x.split('->')[0]], 
                   cellCat=group)==False for x in edgeCols.index]]='lightgray'
        orange_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'orange']
        blue_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'lightblue']
        gray_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'lightgray']
    
    # network plot
    f,ax1 = plt.subplots(1,1,figsize=fsize,dpi=100) 
    #nodes
    if nodeSize == 'betweeness':
        npg = nx.betweenness_centrality(gCol)
        npg=list(npg.values())
        
        nx.draw_networkx_nodes(gCol,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)

    if nodeSize == 'pagerank':
        npg = nx.pagerank(gCol,max_iter=1000, weight=None)
        npg=list(npg.values())  
        
        nx.draw_networkx_nodes(gCol,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)

    if nodeSize == None:
        nx.draw_networkx_nodes(gCol,pos,node_color=color_group,ax=ax1)
    
    #edges
    if group!=None:
        nx.draw_networkx_edges(gCol,pos=pos,edge_color=inter[edgeCols=='lightgray'],
            connectionstyle="arc3,rad=0.15", arrowstyle='<->',
            width=inter[edgeCols=='lightgray']*edge_scale,ax=ax1, edgelist=gray_edges, edge_cmap=graycmp, edge_vmin=-1*np.max(inter), edge_vmax=np.max(inter))
    
    nx.draw_networkx_edges(gCol,pos=pos,edge_color=inter[edgeCols=='lightblue'],
        connectionstyle="arc3,rad=0.15", arrowstyle='<->',
        width=inter[edgeCols=='lightblue']*edge_scale,ax=ax1, edgelist=blue_edges, edge_cmap=cmap3, edge_vmin=-1*np.max(inter), edge_vmax=np.max(inter))
    nx.draw_networkx_edges(gCol,pos=pos,edge_color=inter[edgeCols=='orange'],
        connectionstyle="arc3,rad=0.15", arrowstyle='<->',
        width=inter[edgeCols=='orange']*edge_scale,ax=ax1, edgelist=orange_edges, edge_cmap=cmap4, edge_vmin=-1*np.max(inter), edge_vmax=np.max(inter))
    nx.draw_networkx_labels(gCol,pos_attrs, font_size=12, font_weight='bold', clip_on=False,ax=ax1)

    #color bar
    sm = plt.cm.ScalarMappable(cmap=mymap)
    sm._A = []
    sm.set_clim(-1*np.max(inter), np.max(inter))

    cax = ax1.inset_axes(legend_ax)
    cax.set_xticks([])
    cax.set_yticks([])
    
    cax.axis('off')
    x=plt.colorbar(sm, ax=cax, fraction=0.2)
    x.set_label('diffColoc. score', rotation=270, labelpad=15, size=10, weight='normal')
    x.set_alpha(alpha)

    #assign cell cell interaction scores as edge weights 
    for x in list(gCol.edges):
        gCol[x[0]][x[1]]['weight'] = x_diff.loc[x[0], x[1]]

    to_remove=[(a,b) for a, b, attrs in gCol.edges(data=True) if np.abs(attrs["weight"]) <=thr]
    gCol.remove_edges_from(to_remove)
    
    return gCol

# %%

def colocNW(x_diff,adj, cell_group, group=None, group_cmap='tab20', ncols=20, clist=None, 
            nodeSize=None, legend_ax=[0.7, 0.05, 0.15, 0.2], layout='neato', lab_spacing=9, thr=0, alpha=1, fsize=(8,8), pos=None, 
            edge_scale=1):
    """ (Differential) co-localization network

    Parameters
    ----------
    xdiff : pd.DataFrame
        cell types x cell types data frame of significant 
        scores for each cell cell interaction
    adj : pd.DataFrame
        cell types x cell types adjacency matrix (calculated from the cell cell 
        interaction scores)
    cell_group : dict
        dictionary with niche names as keys and lists of their corresponding cell types as values 
    group : list (default: None)
        list of nodes whose interaction will be highlighted
    group_cmap : str (default: 'tab20')
        name of the color map from which the niche colors will be taken
    ncols : int (default: 20)
        number of colors for the group_cmap
    clist : list (default: None)
        alternatively , one can input a list of niche colors
    nodeSize : str (default: None)
        value that will define the size of the nodes. Options are 'betweeness', 
        'pagerank' , 'signed_betweeness', 'signed_pagerank' (network statistics; 
        the 'signed_*' options size nodes by |log2((pos+eps)/(neg+eps))|, 
        matching nichesphere.tl.compute_network_stats)
    legend_ax : list (default: [0.7, 0.05, 0.15, 0.2])
        legend position in the form [x0, y0, width, height]
    layout : str (default: 'neato')
        name of the layout to be used. Options are 'neato', 'dot', 'kamada_kawai', 
        'spring', 'spectral', 'circular', 'force_atlas2', 'fruchterman_reingold' and 'random')
    lab_spacing : int (default: 9)
        spacing between labels and nodes
    thr : float (default: 0)
        edge weights absolute value must be above this value for the edge to be shown
    alpha : float (default: 1)
        edge transparency (from 0 to 1)
    fsize : tuple
        figure size in the form (x,y)
    pos : dict
        dictionary containing the calculated 2D positions (x, y coordinates) for every node
    edge_scale : float
        factor to scale the edge width

    Returns
    -------
    gCol : nx.Graph
        Graph object with cell cell interaction scores as weights
    Network plot
    """
    
    ## Make color maps
    cmap = plt.cm.RdBu
    cmap3 = cmap(np.arange(cmap.N))
    cmap3[:,-1] = np.linspace(0, alpha, cmap.N)
    c1=cmap3.copy()
    cmap3 = ListedColormap(cmap3)
    
    cmap = plt.cm.RdBu_r
    cmap4 = cmap(np.arange(cmap.N))
    cmap4[:,-1] = np.linspace(0, alpha, cmap.N)
    c2=cmap4.copy()
    cmap4 = ListedColormap(cmap4)

    colors = np.vstack((np.flip(c1[128:256], axis=0), c2[128:256]))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    cmap=mcolors.LinearSegmentedColormap.from_list("WhiteGray",['white','lightgrey'])
    graycmp = cmap(np.arange(cmap.N))
    graycmp[:,-1] = np.linspace(0, alpha-0.2, cmap.N)
    c3=graycmp.copy()
    graycmp = ListedColormap(graycmp)
    
    #cell groups cmap
    cmap = plt.colormaps[group_cmap].resampled(ncols)
    if clist == None:
        cgroup_cmap=[mcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    else:
        cgroup_cmap=clist
    
    gCol=nx.from_pandas_adjacency(adj, create_using=nx.Graph)

    ## Edge thickness
    for x in list(gCol.edges):
        gCol[x[0]][x[1]]['weight'] = np.abs(x_diff.loc[x[0], x[1]])
    
    ## FIX: Map node colors based on exact dictionary key order matching G.nodes()
    niche_keys = list(cell_group.keys())
    niche_color_map = {niche_keys[i]: cgroup_cmap[i] for i in range(len(niche_keys))}
    
    node_to_color = {}
    for niche_id, members in cell_group.items():
        assigned_color = niche_color_map[niche_id]
        for node in members:
            node_to_color[node] = assigned_color
            
    color_group = pd.Series([node_to_color.get(node, '#cccccc') for node in list(gCol.nodes)], index=list(gCol.nodes))

    ### different layouts
    if layout=='neato':
        pos = nx.drawing.nx_agraph.graphviz_layout(gCol,prog='neato')
    if layout=='dot':
        pos = nx.drawing.nx_agraph.graphviz_layout(gCol,prog='dot')
    if layout=='kamada_kawai':
        pos = nx.drawing.kamada_kawai_layout(gCol)
    if layout=='spring':
        pos = nx.drawing.spring_layout(gCol)
    if layout=='spectral':
        pos = nx.drawing.spectral_layout(gCol)
    if layout=='circular':
        pos = nx.drawing.circular_layout(gCol)
    if layout=='force_atlas2':
        pos = nx.drawing.forceatlas2_layout(gCol)
    if layout=='fruchterman_reingold':
        pos = nx.drawing.fruchterman_reingold_layout(gCol)
    if layout=='random':
        pos = nx.drawing.random_layout(gCol)

    if pos!=None:
        pos=pos

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1]+lab_spacing)
    
    to_remove=[(a,b) for a, b, attrs in gCol.edges(data=True) if np.abs(attrs["weight"]) <=thr]
    gCol.remove_edges_from(to_remove)

    ## Edge colors based on diff coloc
    edgeCols=pd.Series(['lightblue' if x_diff.loc[x[0], x[1]]<0 else 'orange' for x in list(gCol.edges)])
    edgeCols.index=[x[0]+'->'+x[1] for x in list(gCol.edges)]
    
    orange_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'orange']
    blue_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'lightblue']

    #normalised scores
    weights = nx.get_edge_attributes(gCol,'weight').values()
    inter=pd.Series(np.abs(pd.Series(list(weights))))
    inter.index=edgeCols.index

    #classify edges by color
    if group!=None:
        edgeCols[[cellCatContained(pair=[x.split('->')[0], x.split('->')[0]], 
                   cellCat=group)==False for x in edgeCols.index]]='lightgray'
        orange_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'orange']
        blue_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'lightblue']
        gray_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'lightgray']
    
    # network plot
    f,ax1 = plt.subplots(1,1,figsize=fsize,dpi=100) 
    #nodes
    if nodeSize == 'betweeness':
        npg = nx.betweenness_centrality(gCol)
        npg=list(npg.values())
        
        nx.draw_networkx_nodes(gCol,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)

    if nodeSize == 'pagerank':
        npg = nx.pagerank(gCol,max_iter=1000, weight=None)
        npg=list(npg.values())  
        
        nx.draw_networkx_nodes(gCol,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)

    if nodeSize == None:
        nx.draw_networkx_nodes(gCol,pos,node_color=color_group,ax=ax1)
    #### signed stats node sizes
    if nodeSize in ('signed_betweeness', 'signed_pagerank'):
        # Rebuild signed weights on the (already thr-filtered) edge set,
        # since gCol currently holds abs(x_diff) for edge thickness.
        G_signed = gCol.copy()
        for x in list(G_signed.edges):
            G_signed[x[0]][x[1]]['weight'] = x_diff.loc[x[0], x[1]]

        G_pos = G_signed.copy()
        G_pos.remove_edges_from(
            [(a, b) for a, b, attrs in G_pos.edges(data=True) if attrs['weight'] <= 0]
        )
        G_neg = G_signed.copy()
        G_neg.remove_edges_from(
            [(a, b) for a, b, attrs in G_neg.edges(data=True) if attrs['weight'] >= 0]
        )

        if nodeSize == 'signed_betweeness':
            bw_pos = nx.betweenness_centrality(G_pos)
            bw_neg = nx.betweenness_centrality(G_neg)
            npg = [np.log2((1e-10 + bw_pos[n]) / (1e-10 + bw_neg[n])) for n in gCol.nodes]

        if nodeSize == 'signed_pagerank':
            pr_pos = nx.pagerank(G_pos)
            pr_neg = nx.pagerank(G_neg)
            npg = [np.log2((1e-10 + pr_pos[n]) / (1e-10 + pr_neg[n])) for n in gCol.nodes]

        npg = np.abs(np.array(npg))  # size by magnitude of imbalance

        nx.draw_networkx_nodes(gCol,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)
    ####
    #edges
    if group!=None:
        nx.draw_networkx_edges(gCol,pos=pos,edge_color=inter[edgeCols=='lightgray'],
            connectionstyle="arc3,rad=0.15", arrowstyle='<->',
            width=inter[edgeCols=='lightgray']*edge_scale,ax=ax1, edgelist=gray_edges, edge_cmap=graycmp, edge_vmin=-1*np.max(inter), edge_vmax=np.max(inter))
    
    nx.draw_networkx_edges(gCol,pos=pos,edge_color=inter[edgeCols=='lightblue'],
        connectionstyle="arc3,rad=0.15", arrowstyle='<->',
        width=inter[edgeCols=='lightblue']*edge_scale,ax=ax1, edgelist=blue_edges, edge_cmap=cmap3, edge_vmin=-1*np.max(inter), edge_vmax=np.max(inter))
    nx.draw_networkx_edges(gCol,pos=pos,edge_color=inter[edgeCols=='orange'],
        connectionstyle="arc3,rad=0.15", arrowstyle='<->',
        width=inter[edgeCols=='orange']*edge_scale,ax=ax1, edgelist=orange_edges, edge_cmap=cmap4, edge_vmin=-1*np.max(inter), edge_vmax=np.max(inter))
    nx.draw_networkx_labels(gCol,pos_attrs, font_size=12, font_weight='bold', clip_on=False,ax=ax1)

    #color bar
    sm = plt.cm.ScalarMappable(cmap=mymap)
    sm._A = []
    sm.set_clim(-1*np.max(inter), np.max(inter))

    cax = ax1.inset_axes(legend_ax)
    cax.set_xticks([])
    cax.set_yticks([])
    
    cax.axis('off')
    x=plt.colorbar(sm, ax=cax, fraction=0.2)
    x.set_label('diffColoc. score', rotation=270, labelpad=15, size=10, weight='normal')
    x.set_alpha(alpha)

    #assign cell cell interaction scores as edge weights 
    for x in list(gCol.edges):
        gCol[x[0]][x[1]]['weight'] = x_diff.loc[x[0], x[1]]

    to_remove=[(a,b) for a, b, attrs in gCol.edges(data=True) if np.abs(attrs["weight"]) <=thr]
    gCol.remove_edges_from(to_remove)
    
    return gCol

# %%
