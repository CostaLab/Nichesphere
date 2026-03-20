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

#%%

def getColocProbs(CTprobs, spotSamples):

    """Get co-localization probabilities 
    (sum across spots of probabilities of each cell type pair being in the same spot)

    Parameters
    ----------
    CTprobs : pd.DataFrame
        Dataframe of cell type probabilities per spot
    spotSamples : pd.Series
        Series indicating the sample to which each spot belongs, with spot ids as index.

    Returns
    -------
    CTcolocalizationP : pd.DataFrame
        concatenated dataframes of cell type pairs co-localization probabilities per sample
    """

    CTcolocalizationP=pd.DataFrame()
    
    for smple in spotSamples.unique():
    
        test=CTprobs.loc[spotSamples.index[spotSamples==smple]]
        CTcoloc_P = pd.DataFrame()
        i=0
        for ct in test.columns:
            w=pd.DataFrame([test[ct]*test[col]/len(test.index) for col in test.columns], index=test.columns).sum(axis=1)
            CTcoloc_P = pd.concat([CTcoloc_P, w], axis=1)
            i=i+1
        
        CTcoloc_P.columns=test.columns
        CTcoloc_P["sample"]=smple
        CTcolocalizationP = pd.concat([CTcolocalizationP, CTcoloc_P])
    
    return CTcolocalizationP

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
    colocPerSample1=pd.DataFrame()
    if(complete==0):
        i=0
        for ct in CTcoloc.columns[range(len(CTcoloc.columns)-1)]:
            x=CTcoloc.iloc[CTcoloc.index==ct,range(len(CTcoloc.columns)-1)]
            for ct2 in CTcoloc.columns[range(i,len(CTcoloc.columns)-1)]:
                probs=pd.DataFrame(x.iloc[:,x.columns==ct2])
                probs.index=CTcoloc["sample"].unique()
                probs.columns=[ct + '-' + ct2]
                colocPerSample1=pd.concat([colocPerSample1, probs], axis=1)
            i=i+1

        colocPerSample1[np.setdiff1d(colocPerSample1.columns, oneCTinteractions)]=colocPerSample1[np.setdiff1d(colocPerSample1.columns, oneCTinteractions)]*2
    else:
        for ct in CTcoloc.columns[range(len(CTcoloc.columns)-1)]:
            x=CTcoloc.iloc[CTcoloc.index==ct,range(len(CTcoloc.columns)-1)]
            for ct2 in CTcoloc.columns[range(len(CTcoloc.columns)-1)]:
                probs=pd.DataFrame(x.iloc[:,x.columns==ct2])
                probs.index=CTcoloc["sample"].unique()
                probs.columns=[ct + '-' + ct2]
                colocPerSample1=pd.concat([colocPerSample1, probs], axis=1)
    
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

def OvsE_coloc_test_adjPval(observedColocProbs, expectedColocProbs, testDistribution_df, cell_types, oneCTinteractions, p=0.05, MTcorrection='fdr_bh'):
    """ Observed vs Expected log2 ratios filtered by p-value obtained from comparing them against a background distribution
    Parameters
    ----------
    observedColocProbs : pd.Series
        observed cell type pair co-localization probabilities in a sample/condition
    expectedColocProbs : pd.Series
        expected cell type pair co-localization probabilities in a sample/condition
    cell_types : list 
        list of cell types (sorted as in observedColocProbs)
    testDistribution_df : pd.DataFrame
        Data frame with distributions of observed vs expected ratios per cell type pair as columns, obtained from random sampling pairs of single cells
        from scRNA data from the same sample/condition (obtained generally from function 'get_PIC_BG_OEratios_DF')
        from the tl module
    oneCTinteractions : list
        list of single cell interactions (celltype-celltype)
    p : float (default: 0.05)
        values must have a p-value lower than this to be considered significant
    MTcorrection : str (default: 'fdr_bh')
        Multiple testing correction to be applied to p-values 
    Returns
    -------
    OvsE_HMdf : pd.DataFrame
        cell types x cell types data frame of log2 observed/expected significant values (scores) for each cell cell interaction
    statsDF : pd.DataFrame
        cell type pairs resulting statistic (log2 O/E score) , p-value and adjusted p-value
    """
    
    OvsE=observedColocProbs/expectedColocProbs
    OvsE_HM=np.log2(OvsE)
    pvals=pd.Series([scipy.stats.percentileofscore(np.log2(testDistribution_df[x]), OvsE_HM[x], kind='weak')/100 for x in OvsE_HM.index])
    
    pvals[pvals>0.5]=1-pvals[pvals>0.5]
    pvals=pvals*2
    pvals_corr=multipletests(pvals, alpha=0.05, method=MTcorrection, maxiter=1, is_sorted=False, returnsorted=False)[1]
    
    OvsE_HM[(pvals_corr>p)]=0
    OvsE_HM[oneCTinteractions]=0
    
    OvsE_HMdf=pd.DataFrame(np.array(OvsE_HM).reshape(-1, len(cell_types)))
    OvsE_HMdf.columns=cell_types
    OvsE_HMdf.index=cell_types
    statsDF=pd.DataFrame({'statistic':list(OvsE_HM), 'pvalue':list(pvals), 'adjusted_pvalue':list(pvals_corr)})
    statsDF.index=observedColocProbs.index
    
    return OvsE_HMdf,statsDF
#%%

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
    cmap = plt.cm.get_cmap(group_cmap, ncols)
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
#%%
