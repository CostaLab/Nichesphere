# %%
import pandas as pd
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

from joblib import Parallel, delayed
import time
from collections import Counter


def get_spot_ct_props(spot_cell_props, sc_ct):
    """ Get cell type proportions per spot by summing the probabilities of cells of the same 
    kind in each spot

    Parameters
    ----------
    spot_cell_props : pd.DataFrame
        Dataframe containing probabilities of mapping each cell to each spot (spot ids = index , 
        cell ids = columns)
    sc_ct : pd.Series
        Series of cell type per cell with cell ids as indexes

    Returns
    -------
    spot_mapped_cts : pd.DataFrame
        Dataframe containing probabilities of mapping each cell type to each spot (spot ids = index , 
        cell types = columns)
    """
    arr=[np.array([np.sum(np.array(spot_cell_props.iloc[:, location][np.argwhere(sc_ct == cluster).flatten()])) for cluster in sc_ct.unique()]) for location in range(spot_cell_props.shape[1])]
    spot_mapped_cts=pd.DataFrame(arr, columns=sc_ct.unique(), index=spot_cell_props.columns)
    return spot_mapped_cts

# %%
def cellCatContained(pair, cellCat):
    """Generate boolean element indicating whether the specified element (cell pair) is contained in 
    a list (category)

    Parameters
    ----------
    pair : str
        Evaluated element (cell type pair usually in this context)
    cellCat : list
        Category to test (EG: list containing a cell type category)

    Returns
    -------
    True or False
    
    """
    
    contained=[cellType in pair for cellType in cellCat]
    return True in contained
# %%
def cells_niche_colors(CTs, niche_colors, niche_dict):
    """ Make dataframe of cell types with their corresponding niches and colors

    Parameters
    ----------
    CTs : pd.Series or list
        list of cell types
    niche_colors : pd.Series
        Series of colors in hexadecimal format with niche names as index
    niche_dict : dictionary
        Dictionary with niche names as keys and cell types as elements

    Returns
    -------
    niche_df : pd.DataFrame
        Dataframe of cell types with their corresponding niches and colors
    
    """
    niche_df=pd.DataFrame(CTs, columns=['cell'])
    niche_df['niche']=niche_colors.index[0]
    niche_df['color']=niche_colors[0]
    for key in list(niche_dict.keys()):
        niche_df['niche'][[c in niche_dict[key] for c in niche_df.cell]]=key
        niche_df['color'][niche_df['niche']==key]=niche_colors[key]
    niche_df.index=niche_df.cell
    niche_df.niche=niche_df.niche.astype('category')
    return niche_df
#%%
def unique(array):
    """get unique elements in array without re-sorting

    Parameters
    ----------
    array : np.array

    Returns
    -------
    uniq[index.argsort()] : np.array of unique elements of the original array in original order
    """
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

#%%
def pval_filtered_HMdf(testDF, oneCTinteractions, p, cell_types):
    """ Get dataframe for scores heatmap from dataframe obtained from diffColoc_test()
    
    Parameters
    ----------
    testDF : pd.DataFrame
        dataframe with cell pairs as indexes and columns 'statistic' and 'p-value'
        (often resulting from diffColoc_test())
    oneCTinteractions : list
        list of single cell interactions (celltype-celltype)
    p : float
        threshold p-value to filter
    cell_types : pd.Series or list
        list/series of cell types
    
    Returns
    -------
    testHM : pd.DataFrame
        Dataframe of cell types x cell types with significant 'statistic' values
    """
    testHM=testDF.statistic.copy()
    testHM[testDF['p-value']>p]=0
    testHM[oneCTinteractions]=0
    # Reshape into data frame
    testHM=pd.DataFrame(np.array(testHM).reshape(-1, len(cell_types)))
    testHM.columns=cell_types
    testHM.index=cell_types
    testHM=testHM.apply(pd.to_numeric)
    
    return testHM
#%%

def PIC_BGdoubletsOEratios(adata_singlets, annot_col):
    """Generate distribution of O/E ratios for colocalization probabilities of cell type pairs in randomly generated doublets (background dist)
    for tests in PIC-seq data
    
    Parameters
    ----------
    adata_singlets : AnnData
        anndata object containing singlets (scRNA-seq) data
    annot_col : str
        name of the cell type annotation column in the anndata object obs

    Returns
    -------
    OEratios : pd.Series
        distribution of O/E ratios for colocalization probabilities of cell type pairs in randomly generated doublets
    """
        
    rdf=pd.DataFrame(adata_singlets.obs[annot_col])
    rdf.columns=['annot']
    rdf.index=adata_singlets.obs.index
    rdf['pair']=''
    
    ## Get random singlets pairs
    pairNums=[i for i in range(int(np.round(adata_singlets.obs.shape[0]//2))) for _ in range(2)]
    pairNumsIdx=random.sample(list(adata_singlets.obs.index), len(pairNums))
    rdf.pair[pairNumsIdx]=pairNums

    pairCounts=[rdf.annot[rdf.pair==i][0]+'-'+rdf.annot[rdf.pair==i][1] for i in rdf.pair.value_counts().index[rdf.pair.value_counts()==2]]
    
    ## Expected probabilities of cell type pairs
    probs=rdf.annot[[i!='' for i in rdf.pair]].value_counts()/rdf.annot[[i!='' for i in rdf.pair]].value_counts().sum()

    pairProbs=pd.DataFrame()
    for x in probs:
        pairProbs=pd.concat([pairProbs, pd.DataFrame(x*probs)])
        
    pci=[]
    for x in probs.index:
        pci.append((x+'-'+probs.index.astype(str)).tolist())    
    pairProbs.index=[item for sublist in pci for item in sublist]

    ## Observed probabilities of cell type pairs
    pairProbsO=pd.Series(pairCounts).value_counts()/pd.Series(pairCounts).value_counts().sum()
    ## O/E ratios
    OEratios=pairProbsO/pairProbs['count'][pairProbsO.index]
    return OEratios


#%%

def PIC_BGdoubletsOEratios_parallel(seed, adata, annot_col, target_pairs):
    
    """Generate set of O/E ratios for colocalization probabilities of cell type pairs based on randomly paired single cells.
    n runs of this function generate distributions of n values for each cell type pair, which can be used for tests in PIC-seq data or datasets with one sample
    
    Parameters
    ----------
    seed : int
        seed for generating random distributions
    adata : AnnData
        anndata object containing singlets (scRNA-seq) data
    annot_col : str
        name of the cell type annotation column in the anndata object obs
    target_pairs : list or pd.DataFrame
        list of cell type pairs to be evaluated

    Returns
    -------
    oe_ratios : pd.Series
        set of O/E ratios for colocalization probabilities of cell type pairs in randomly paired single cells
    """
    
    # Initialize a local random generator with the specific seed
    random.seed(seed)
    # 1. Shuffle to randomize the order
    my_list=list(adata.obs[annot_col])
    random.shuffle(my_list)
    # 2. Pair them up and format into strings
    it = iter(my_list)
    observed_pairs = [f"{a}-{b}" for a, b in zip(it, it)]

    # 3. Calculate Observed Frequencies
    obs_counts = Counter(observed_pairs)
    total_gen = len(observed_pairs)
    
    # 4. Calculate Expected Frequencies
    # Expected = P(Item1) * P(Item2 | Item1) * Total Pairs
    item_counts = Counter(my_list)
    n = len(my_list)

    def calculate_ratio(pair_str):
        try:
            a, b = pair_str.split('-')
            # Probability of A then B
            p_a = item_counts[a] / n
            p_b = (item_counts[b] - (1 if a == b else 0)) / (n - 1)
            expected = p_a * p_b * total_gen
            
            observed = obs_counts.get(pair_str, 0)
            return observed / expected if expected > 0 else 0.0
        except:
            return 0.0

    # 5. Create the Pandas Series
    oe_ratios = pd.Series(
        {pair: calculate_ratio(pair) for pair in target_pairs},
        name="OE_Ratio"
    )
    
    #t=pd.Series(random_strings)
    return oe_ratios
#%%

def get_PIC_BG_OEratios_DF(adata, annot_col, target_pairs, nreps=1000, njobs=10):
    """Generate nreps sets of O/E ratios for colocalization probabilities of cell type pairs based on randomly paired single cells via the PIC_BGdoubletsOEratios_parallel function.
    Depending on nreps this function generate that number of sets of O/E values for each cell type pair, which can be used for tests based on empirical p-values in PIC-seq data or datasets with one sample
    
    Parameters
    ----------
    adata : AnnData
        anndata object containing singlets (scRNA-seq) data
    annot_col : str
        name of the cell type annotation column in the anndata object obs
    target_pairs : list or pd.DataFrame
        list of cell type pairs to be evaluated
    nreps : int (default: 1000)
        number of sets of O/E values to obtain for each cell type pair
    njobs : int (default: 10)
        number of parallel jobs to be executed

    Returns
    -------
    df : pd.DataFrame
        Data frame of nreps rows , which will be sets of O/E ratios for colocalization probabilities of cell type pairs in randomly paired single cells
    """
    t = time.time()
    res=Parallel(n_jobs=njobs)(delayed(PIC_BGdoubletsOEratios_parallel)(seed=i, adata=adata, annot_col=annot_col, target_pairs=target_pairs) for i in range(nreps))
    df = pd.concat(res, axis=1).T
    print(time.time() - t)
    return df
#%%

def getExpectedColocProbsFromSCs(sc_adata, sample, cell_types, sc_data_sampleCol, sc_adata_annotationCol):
    """Compute the expected probability of each cell type pair to occur in a specific condition by multiplying cell type proportions from 
    the single cell data

    Parameters
    ----------
    sc_adata : AnnData
        anndata object containing singlets (scRNA-seq) data
    sample : str
        name of the sample/condition to be tested
    cell_types : pd.Series or list
        list/series of cell types 
    sc_data_sampleCol : str
        name of the column in the obs of the anndata object where the sample/condition is indicated 
    sc_adata_annotationCol : str
        name of the cell type column in the obs of the anndata object

    Returns
    -------
    scCTpairsProbs : pd.DataFrame
        Dataframe of expected co-localization probabilities per cell type pair.
        Cell type pairs as indexes and a 'count' column
    """
    
    scCTprops=sc_adata.obs[sc_adata_annotationCol][sc_adata.obs[sc_data_sampleCol]==sample].value_counts()[cell_types]/sc_adata.obs[sc_adata_annotationCol][sc_adata.obs[sc_data_sampleCol]==sample].value_counts().sum()
    scCTpairsProbs=pd.DataFrame()
    
    for x in scCTprops:
        scCTpairsProbs=pd.concat([scCTpairsProbs, pd.DataFrame(x*scCTprops)])
        
    pci=[]
    for x in scCTprops.index:
        pci.append((x+'-'+scCTprops.index.astype(str)).tolist())    
    scCTpairsProbs.index=[item for sublist in pci for item in sublist]
    return scCTpairsProbs
#%%
    
def get_pairCatDFdir(niches_df):
    
    """Get dataframe of cell pair to niche pair correspondence

    Parameters
    ----------
    niches_df : pd.Dataframe
        Dataframe of cells, corresponding niche and corresponding niche color

    Returns
    -------
    pairCatDFdir : pd.DataFrame
        dataframe of cell pairs and corresponding niche pairs
    """
    pairsDir=[]
    for ct in niches_df.cell[range(len(niches_df.cell))]:
        for ct2 in niches_df.cell[range(len(niches_df.cell))]:
            pairsDir.append(ct+'->'+ct2)
    pairCatDFdir=pd.DataFrame(pairsDir, columns=['cell_pairs'])
    
    pairCatDFdir['niche_pairs']=''
    for clust in np.sort(niches_df.niche.unique()):
        pairCatDFdir['niche_pairs'][[cellCatContained(pair=p, cellCat=niches_df.cell[niches_df.niche==clust]) for p in pairCatDFdir.cell_pairs]]=clust+'->'+clust

    for comb in list(itertools.permutations(list(niches_df.niche.unique().sort_values()), 2)):
        pairCatDFdir['niche_pairs'][[(p.split('->')[0] in niches_df.cell[niches_df.niche==comb[0]]) & (p.split('->')[1] in niches_df.cell[niches_df.niche==comb[1]]) for p in pairCatDFdir.cell_pairs]]=comb[0]+'->'+comb[1]

    return pairCatDFdir
#%%

def processCTKRoutput(ctkrTbl):
    """Remove suffixes '|L' , '|R' and '|TF' from cell - cell communication (CrossTalkeR) tables

    Parameters
    ----------
    ctkrTbl : pd.DataFrame
        Condition specific cell - cell communication table (output from CrossTalkeR)
    Returns
    -------
    ctkrTbl : pd.DataFrame
        Same cell - cell communication table without the suffixes mentioned above
    """
    ctkrTbl['gene_A']=ctkrTbl['gene_A'].str.replace('|L', '')
    ctkrTbl['gene_A']=ctkrTbl['gene_A'].str.replace('|R', '')
    ctkrTbl['gene_B']=ctkrTbl['gene_B'].str.replace('|R', '')
    ctkrTbl['gene_B']=ctkrTbl['gene_B'].str.replace('|TF', '')

    ctkrTbl['allpair']=ctkrTbl['allpair'].str.replace('|R', '')
    ctkrTbl['allpair']=ctkrTbl['allpair'].str.replace('|L', '')
    ctkrTbl['allpair']=ctkrTbl['allpair'].str.replace('|TF', '')
    return ctkrTbl

#%%
def getColocFilter(pairCatDF, adj, oneCTints):
    """Get filtering object indicating which cell pairs are differentially co-localized

    Parameters
    ----------
    pairCatDF : pd.DataFrame
        dataframe of cell pairs and corresponding niche pairs
    adj : pd.DataFrame
        cell types x cell types adjacency matrix (calculated from the cell cell 
        interaction scores)
    oneCTints : list
        list of single cell interactions (celltype->celltype)
    Returns
    -------
    colocFilt : pd.DataFrame
        Dataframe with cell type pairs in the form 'celltype->celltype' as index and a 'filter' column indicating which cell pairs 
        are differentially co-localized with a 1 (or 0 if they are not)
    """
    colocFilt=pd.DataFrame(pairCatDF.cell_pairs, columns=['cell_pairs'], 
                       index=pairCatDF.cell_pairs)
    colocFilt['filter']=0

    for i in pairCatDF.cell_pairs:
        colocFilt['filter'][i]=adj.loc[i.split('->')[1],i.split('->')[0]]
    
    colocFilt['filter'][oneCTints]=1
    colocFilt['filter'][colocFilt['filter']>0]=1
    colocFilt=pd.DataFrame(colocFilt['filter'], index=colocFilt.index, columns=['filter'])
    return colocFilt

#%%
def assign_properties(g, communities, colors, pos=None, node_coord_sf=200, simmilarity_weights=False, g_unsigned=None, alpha=1):
    """Assign properties to network g to visualize with gravis interactive networks

    Parameters
    ----------
    g : nx.Graph
        Graph object with cell cell interaction scores as weights
    communities : list of sets 
        cell type communities
    colors : list
        list of colors (one per community)
    pos : dict (default: None)
        dictionary of node positions in a layout. Keys are cell types and values are arrays of x,y coordinates
    node_coord_sf : int (default: 200)
        node coordinate scale factor to adapt to gravis
    simmilarity_weights : bool (default: False)
        whether or not to have the simmilarity based weights (from the unsigned network)
    g_unsigned : nx.Graph (default: None)
        Unsigned network generated directly from the adjacency matrix derived from the differential co-localization scores 
        (with nx.from_pandas_adjacency)
    alpha : float (default: 1)
        edge transparency parameter (from 0 to 1)
    Returns
    -------
    g : nx.Graph
        Graph object with cell cell interaction scores as weights and added properties 
        (different options for node size scaling, like 'size_betweeness', 'size_pagerank' and 'size_pagerank_uw', node shape, color, x and y coordinates, 
        options for edge weights, like 'weight_signed' (from co-localization/communication statistical test), 'weight_abs' (absolute values without signs), 'weight_exp' (exponentially scaled) and 'weight_simmilarity',
        edge color and size). Edges and nodes visualization options can be managed interactively with th gravis library
    """
    
    ## Create color map for gravis interactive network (to move nodes around and change label sizes)
    alpha=alpha
    # Choose colormap for 1st half
    cmap = plt.cm.RdBu
    # Get the colormap colors
    cmap3 = cmap(np.arange(cmap.N))
    # Set alpha (transparency)
    cmap3[:,-1] = np.linspace(0, alpha, cmap.N)
    c1=cmap3.copy()
    # Create new colormap
    cmap3 = mcolors.ListedColormap(cmap3)
        
    # Choose colormap for 2nd half
    cmap = plt.cm.RdBu_r
    # Get the colormap colors
    cmap4 = cmap(np.arange(cmap.N))
    # Set alpha (transparency)
    cmap4[:,-1] = np.linspace(0, alpha, cmap.N)
    c2=cmap4.copy()
    # Create new colormap
    cmap4 = mcolors.ListedColormap(cmap4)

    #Mixed color map
    colors_edges = np.vstack((np.flip(c1[128:256], axis=0), c2[128:256]))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors_edges)

    #get max. absolute weight in network to constraint color map
    max_col=np.max(np.abs(list(nx.get_edge_attributes(g,'weight').values())))

    #constraint color map
    norm = mcolors.Normalize(vmin=-max_col, vmax=max_col)

    # Centrality calculation
    node_centralities_bet = nx.betweenness_centrality(g)
    node_pr_uw = nx.pagerank(g, max_iter=1000, weight=None)
    

    # Graph properties
    g.graph['node_border_size'] = 1.5
    g.graph['node_border_color'] = 'white'
    g.graph['edge_opacity'] = 0.9

    # Node properties: Size by centrality, color by community
    for node_id in g.nodes:
        node = g.nodes[node_id]
        node['size_betweeness'] = 10 + node_centralities_bet[node_id] * 100
        node['size_pagerank_uw'] = 10 + node_pr_uw[node_id] * 100
        node['shape'] = 'circle'
        
        for community_counter, community_members in enumerate(communities):
            if node_id in community_members:
                break
        node['color'] = colors[community_counter % len(colors)]

    if pos!=None:
        # Add coordinates as node annotations that are recognized by gravis
        for name, (x, y) in pos.items():
            node = g.nodes[name]
            node['x'] = x*node_coord_sf
            node['y'] = y*node_coord_sf
    
    # Edge properties: Size and color by weight
    for edge_id in g.edges:
        edge =  g.edges[edge_id]
        edge['color']=mcolors.to_hex(mymap(norm(edge['weight'])))

        #### weights for plotting
        edge['weight_signed']=edge['weight']
        edge['weight_abs']=np.abs(edge['weight'])
        edge['weight_exp']=np.exp(edge['weight'])
    if simmilarity_weights==True:
        for edge_id in g.edges:
            #### simmilarity based weights
            edge =  g.edges[edge_id]
            edge['weight_simmilarity']=g_unsigned.edges[edge_id]['weight']

        node_pr = nx.pagerank(g, max_iter=1000, weight='weight_simmilarity')

        for node_id in g.nodes:
            node = g.nodes[node_id]
            node['size_pagerank'] = 10 + node_pr[node_id] * 100
            
            
#%%