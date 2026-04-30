# %%
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import sklearn
from matplotlib.colors import ListedColormap

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
def calculate_LR_CT_pair_scores_dir(ccommTable, LRscoresCol):
    """Get cell communication scores per cell type pair per LR pair by summing that LR pair scores for that cell type pair. 
    
    Parameters
    ----------
    ccommTable : pd.DataFrame
        Condition specific cell - cell communication table (output from CrossTalkeR)
    LRscoresCol : str
        name of the ccommTable column where the LR scores are (usually 'MeanLR' for CrossTalkeR)
    Returns
    -------
    scores : pd.Series
        cell communication scores per cell type / LR pair (allpair in CrossTalkeR)
    """
    
   
    scores = ccommTable[LRscoresCol].groupby(ccommTable['allpair']).sum()
    return scores
#%%
def lr_ctPairScores_perCat_dir(ccommTable, db, dbCatCol, dbMatchCol, ccommMatchCol, ccommLRscoresCol, oneCTinteractions, condition, pairCatDF):
    """Calculate cell communication scores per ligand category from a database
    
    Parameters
    ----------
    ccommTable : pd.DataFrame
        Condition specific cell - cell communication table (output from CrossTalkeR)
    db : pd.DataFrame
        database table with information about element (ligand or receptor or both) category (eg: biological processes)
    dbCatCol : str
        column in the database table containing the category (eg: biological process) with which the ligands/receptors/LR pairs are associated
    dbMatchCol : str
        column in the database table containing the elements (ligands/receptors/LR pairs) associated to the categories
    ccommMatchCol : str
        column in the ccommTable table containing the elements (ligands/receptors/LR pairs) associated to the categories
    ccommLRscoresCol : str
        name of the ccommTable column where the LR scores are (usually 'MeanLR' for CrossTalkeR)
    oneCTinteractions : list
        list of single cell interactions (celltype@celltype)
    condition : str
        name of the analyzed condition (a column with this string will be added to the resulting table)
    pairCatDF : pd.DataFrame
        dataframe of cell pairs and corresponding co-localization niche pairs
    Returns
    -------
    CTpairScores_byCat : pd.DataFrame
        cell communication table with new columns 'cell_pairs', 'niche_pairs', 'LRcat' and 'condition'
    """
    pairCatDF.index=pairCatDF.cell_pairs
    db[dbMatchCol]=db[dbMatchCol].str.lower()
    ccommTable=ccommTable.iloc[[not(x in oneCTinteractions) for x in ccommTable['cellpair']],:]
    CTpairScores_byCat=pd.DataFrame()
    for cat in db[dbCatCol].unique():    
    
        ccommScores_plt=pd.DataFrame(calculate_LR_CT_pair_scores_dir(ccommTable=ccommTable[[x in db[dbMatchCol][db[dbCatCol]==cat].tolist() for x in ccommTable[ccommMatchCol].str.lower().tolist()]], LRscoresCol=ccommLRscoresCol))
        
        ct1=[x[0] for x in ccommScores_plt.index.str.split('/')]
        ct2=[x[1].split('@')[1] for x in ccommScores_plt.index.str.split('/')]
        ccommScores_plt['cellpair']=[ct1[i]+'->'+ct2[i] for i in range(len(ccommScores_plt.index))]
        
        boxplotDF=pairCatDF.loc[ccommScores_plt.cellpair,:]
        boxplotDF.index=ccommScores_plt.index
        boxplotDF['LRscores']=ccommScores_plt[ccommLRscoresCol]
        boxplotDF['LRcat']=cat
        CTpairScores_byCat=pd.concat([CTpairScores_byCat, boxplotDF])
    
    CTpairScores_byCat['condition']=condition
        
    return CTpairScores_byCat


#%%

def equalizeScoresTables(ctrlTbl, expTbl, ctrlCondition, expCondition):
    """Makes communication score tables contain the same interactions through adding 0s to be compared
    
    Parameters
    ----------
    ctrlTbl : pd.DataFrame
        Condition specific (control) processed cell - cell communication table
    expTbl : pd.DataFrame
        Condition specific (exp) processed cell - cell communication table
    ctrlCondition : str
        name of the control condition (a column with this string will be added to the resulting table)
    expCondition : str
        name of the exp condition (a column with this string will be added to the resulting table)
    
    Returns
    -------
    ctrlTbl,expTbl
    ctrlTbl : pd.DataFrame
        control cell communication table with all interactions
    expTbl : pd.DataFrame
        exp cell communication table with all interactions
    """
    t=ctrlTbl.loc[np.setdiff1d(ctrlTbl.index, expTbl.index)]
    t.LRscores=0
    t.condition=expCondition
    expTbl=pd.concat([expTbl, t])

    t=expTbl.loc[np.setdiff1d(expTbl.index, ctrlTbl.index)]
    t.LRscores=0
    t.condition=ctrlCondition
    ctrlTbl=pd.concat([ctrlTbl, t])

    return ctrlTbl, expTbl
#%%
def diffCcommStats(c1CTpairScores_byCat, c2CTpairScores_byCat, cellCatCol):
    """Differential cell communication per LR category (eg: biological process)
    
    Parameters
    ----------
    c1CTpairScores_byCat : pd.DataFrame
        exp cell communication table with all interactions
    c2CTpairScores_byCat : pd.DataFrame
        control cell communication table with all interactions
    cellCatCol : str
        name of the column in the communication tables containing the cell pair grouping we would like to compare
        (eg: 'niche_pairs', 'cell_pairs')
    Returns
    -------
    diffCommTable : pd.DataFrame
        dataframe of Wilcoxon statictics and p-values for each cell pair grouping in each LR category (eg: biological process)
        columns are named 'wilcoxStat', 'wilcoxPval', 'cellCat' and 'LRcat'
    
    """
    diffCommTable=pd.DataFrame()
    for LRcat in c1CTpairScores_byCat.LRcat.unique():
        tmp=pd.DataFrame([scipy.stats.ranksums(c1CTpairScores_byCat.LRscores[(c1CTpairScores_byCat[cellCatCol]==cat) & (c1CTpairScores_byCat.LRcat==LRcat)], c2CTpairScores_byCat.LRscores[(c2CTpairScores_byCat[cellCatCol]==cat) & (c2CTpairScores_byCat.LRcat==LRcat)]).statistic for cat in c1CTpairScores_byCat[cellCatCol].unique()],
                        columns=['wilcoxStat'])
        tmp['wilcoxPval']=[scipy.stats.ranksums(c1CTpairScores_byCat.LRscores[(c1CTpairScores_byCat[cellCatCol]==cat) & (c1CTpairScores_byCat.LRcat==LRcat)], c2CTpairScores_byCat.LRscores[(c2CTpairScores_byCat[cellCatCol]==cat) & (c2CTpairScores_byCat.LRcat==LRcat)]).pvalue for cat in c1CTpairScores_byCat[cellCatCol].unique()]
        tmp['cellCat']=c1CTpairScores_byCat[cellCatCol].unique()
        tmp['LRcat']=LRcat
        diffCommTable=pd.concat([diffCommTable, tmp])
    
    return diffCommTable
#%%

def plotDiffCcommStatsHM(diffCommTable, min_pval, vmin=None, vmax=None):
    """Plot heatmap of differential cell communication statistics
    
    Parameters
    ----------
    diffCommTable : pd.DataFrame
        dataframe of Wilcoxon statictics and p-values for each cell pair grouping in each LR category
        (obtained with diffCcommStats function)
    min_pval : float
        scores with p-values higher than this won't be shown
    vmin : float (default: None)
        minimum value for the color map
    vmax : float (default: None)
        maximum value for the color map

    Returns
    -------
    x_hm : pd.DataFrame
        differential communication scores dataframe of cell pair groupings x LR categories
    plot : sns.matrix.ClusterGrid
        seaborn clustermap as an object
    """
    x=pd.Series(diffCommTable.wilcoxStat)
    ## Remove non significant values and NaNs
    x[[i>min_pval for i in np.array(diffCommTable.wilcoxPval)]]=0
    x[np.isnan(x)]=0
    ## Make dataframe to plot heatmap
    x_hm=pd.DataFrame(np.array(x).reshape(-1, len(diffCommTable.cellCat.unique())))
    x_hm.columns=diffCommTable.cellCat.unique()
    x_hm.index=diffCommTable.LRcat.unique()
    ## Plot heatmap
    sns.set_theme(font_scale=1.5)
    plot=sns.clustermap(x_hm, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax)
    plt.setp(plot.ax_heatmap.yaxis.get_majorticklabels(), rotation=1)
    return x_hm, plot

#%%
def getDiffComm(diffCommTbl, pairCatDF, ncells, cat):
    """get the differential communication scores for a specific LR category
    
    Parameters
    ----------
    diffCommTbl : pd.DataFrame
        differential communication scores dataframe of cell pairs (or groupings) x LR categories
        (obtained with plotDiffCcommStatsHM function)
    pairCatDF : pd.DataFrame
    ncells : int
        number of cells or groups
    cat : str 
        LR category to be tested
    Returns
    -------
    x_chem : pd.DataFrame
        cells x cells or groups x groups dataframe of differential communication scores for a specific LR category
    """
    x=pd.DataFrame(pairCatDF.cell_pairs)
    x['wilcoxStat']=0.0
    x.index=pairCatDF.cell_pairs

    
    for i in diffCommTbl.columns:
        x.loc[i, 'wilcoxStat']=diffCommTbl[i][cat]

    
    x=pd.Series(x.wilcoxStat)
    x_chem=pd.DataFrame(np.array(x).reshape(-1, ncells))
    x_chem.columns=unique([x.split('->')[0] for x in pairCatDF.cell_pairs])
    x_chem.index=unique([x.split('->')[0] for x in pairCatDF.cell_pairs])

    return x_chem

#%%

def catNW(x_chem,colocNW, cell_group, group=None, group_cmap='tab20', ncols=20, color_group=None, plot_title='', 
          clist=None, nodeSize=None, legend_ax=[0.7, 0.05, 0.15, 0.2], layout='neato', thr=0, fsize=(8,8), alpha=1, lab_spacing=7, edge_scale=1, pos=None):    
    """(Differential) communication network
    
    Parameters
    ----------
    xchem : pd.DataFrame
        cell types x cell types data frame of significant 
        scores for each cell cell interaction (for a specific LR category)
    colocNW : nx.Graph
        Graph object with (differential) co-localization scores as weights 
    cell_group : dict
        dictionary with niche names as keys and lists of their corresponding cell types as values 
    group : list (default: None)
        list of nodes whose interaction will be highlighted
    group_cmap : str (default: 'tab20')
        name of the color map from which the niche colors will be taken
    ncols : int (default: 20)
        number of colors for the group_cmap
    color_group : list
        list of colors (one color per node)
    plot_title : str (default: '')
        plot title to be shown at the top of the network plot
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
    G : nx.Graph
        Graph object with (differential) cell communication scores as weights
    Network plot
    """

    # Choose colormap
    cmap = plt.cm.RdBu
    # Get the colormap colors
    cmap3 = cmap(np.arange(cmap.N))
    # Set alpha (transparency)
    cmap3[:,-1] = np.linspace(0, alpha, cmap.N)
    c1=cmap3.copy()
    # Create new colormap
    cmap3 = ListedColormap(cmap3)
    
    # Choose colormap
    cmap = plt.cm.RdBu_r
    # Get the colormap colors
    cmap4 = cmap(np.arange(cmap.N))
    # Set alpha (transparency)
    cmap4[:,-1] = np.linspace(0, alpha, cmap.N)
    c2=cmap4.copy()
    # Create new colormap
    cmap4 = ListedColormap(cmap4)

    colors = np.vstack((np.flip(c1[128:256], axis=0), c2[128:256]))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    cmap=mcolors.LinearSegmentedColormap.from_list("WhiteGray",['white','lightgrey'])
    # Get the colormap colors
    graycmp = cmap(np.arange(cmap.N))
    # Set alpha (transparency)
    graycmp[:,-1] = np.linspace(0, alpha-0.2, cmap.N)
    c3=graycmp.copy()
    # Create new colormap
    graycmp = ListedColormap(graycmp)
    
    #cell group cmap
    cmap = plt.colormaps[group_cmap].resampled(ncols)
    if clist == None:
        cgroup_cmap=[mcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    else:
        cgroup_cmap=clist
    
    ###
    # create comm network
    G=nx.Graph()
    G.add_nodes_from(colocNW)
    G.add_edges_from(colocNW.edges())
    G=G.to_directed()
    ###

       
    ## Node color groups
    if color_group is None:
        color_group=pd.Series(list(G.nodes))
        i=0
        for k in list(cell_group.keys()):
            color_group[[cellCatContained(pair=p, cellCat=cell_group[k]) for p in color_group]]=cgroup_cmap[i]
            i=i+1
        
    ## Edge thickness
    for x in list(G.edges):
        G[x[0]][x[1]]['weight'] = x_chem.loc[x[0], x[1]]
    
    ### different layouts
    if layout=='neato':
        pos = nx.drawing.nx_agraph.graphviz_layout(G,prog='neato')
    if layout=='dot':
        pos = nx.drawing.nx_agraph.graphviz_layout(G,prog='dot')
    if layout=='kamada_kawai':
        pos = nx.drawing.kamada_kawai_layout(G)
    if layout=='spring':
        pos = nx.drawing.spring_layout(G)
    if layout=='spectral':
        pos = nx.drawing.spectral_layout(G)
    if layout=='circular':
        pos = nx.drawing.circular_layout(G)
    if layout=='force_atlas2':
        pos = nx.drawing.forceatlas2_layout(G)
    if layout=='fruchterman_reingold':
        pos = nx.drawing.fruchterman_reingold_layout(G)
    if layout=='random':
        pos = nx.drawing.random_layout(G)

    if pos!=None:
        pos=pos

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1]+lab_spacing)


    #to_remove=[(a,b) for a, b, attrs in G.edges(data=True) if attrs["weight"] == 0]
    to_remove=[(a,b) for a, b, attrs in G.edges(data=True) if np.abs(attrs["weight"]) <= thr]
    G.remove_edges_from(to_remove)
    f,ax1 = plt.subplots(1,1,figsize=fsize,dpi=100) 

    ###
    weights=nx.get_edge_attributes(G,'weight').values()
    ## Edge colors based on diff comm
    edgeCols=pd.Series(['lightblue' if x_chem.loc[x[0], x[1]]<0 else 'orange' for x in list(G.edges)])
    edgeCols.index=[x[0]+'->'+x[1] for x in list(G.edges)]
    
    orange_edges = [(u,v) for u,v in G.edges if edgeCols[u+'->'+v] == 'orange']
    blue_edges = [(u,v) for u,v in G.edges if edgeCols[u+'->'+v] == 'lightblue']
    
    inter=pd.Series(np.abs(pd.Series(list(weights))))
    inter.index=edgeCols.index

    ####### NEW #######
    if group!=None:
        edgeCols[[cellCatContained(pair=[x.split('->')[0], x.split('->')[0]], 
                   cellCat=group)==False for x in edgeCols.index]]='lightgray'
        orange_edges = [(u,v) for u,v in G.edges if edgeCols[u+'->'+v] == 'orange']
        blue_edges = [(u,v) for u,v in G.edges if edgeCols[u+'->'+v] == 'lightblue']
        gray_edges = [(u,v) for u,v in G.edges if edgeCols[u+'->'+v] == 'lightgray']
    ###################
    
    ###
    
    if nodeSize == 'betweeness':
        ## pagerank sized nodes
        npg = nx.betweenness_centrality(G)
        npg=list(npg.values())

        ## edges positions
        pos_edges=pos        
        
        nx.draw_networkx_nodes(G,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)

    if nodeSize == 'pagerank':
        ## pagerank sized nodes
        npg = nx.pagerank(G,max_iter=1000, weight=None)
        npg=list(npg.values())

        ## edges positions
        pos_edges=pos        
        
        nx.draw_networkx_nodes(G,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)

    if nodeSize == None:
        pos_edges=pos


        nx.draw_networkx_nodes(G,pos,node_color=color_group,ax=ax1)
    
    ##########################################
    if group!=None:
        nx.draw_networkx_edges(G,pos=pos,edge_color=inter[edgeCols=='lightgray'],
            connectionstyle="arc3,rad=0.15", arrowstyle='<->',
            width=inter[edgeCols=='lightgray']*edge_scale,ax=ax1, edgelist=gray_edges, edge_cmap=graycmp, edge_vmin=-1*np.max(inter), edge_vmax=np.max(inter))
    ##########################################
    nx.draw_networkx_edges(G,pos=pos_edges,edge_color=inter[edgeCols=='lightblue'],
            connectionstyle="arc3,rad=0.15",
            width=inter[edgeCols=='lightblue']*edge_scale,ax=ax1, edgelist=blue_edges, edge_cmap=cmap3,edge_vmin=-1*np.max(inter), 
            edge_vmax=np.max(inter), arrowsize=20)
    nx.draw_networkx_edges(G,pos=pos_edges,edge_color=inter[edgeCols=='orange'],
            connectionstyle="arc3,rad=0.15",
            width=inter[edgeCols=='orange']*edge_scale,ax=ax1, edgelist=orange_edges, edge_cmap=cmap4,edge_vmin=-1*np.max(inter), 
            edge_vmax=np.max(inter), arrowsize=20)
    nx.draw_networkx_labels(G,pos_attrs,verticalalignment='bottom',
        font_size=12,clip_on=False,ax=ax1, font_weight='bold')
    f.suptitle(plot_title)
    
    sm = plt.cm.ScalarMappable(cmap=mymap)
    sm._A = []
    sm.set_clim(-1*np.max(inter), np.max(inter))

    cax = ax1.inset_axes(legend_ax)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.patch.set_alpha(alpha)
    cax.axis('off')
    x=plt.colorbar(sm, ax=cax, fraction=0.2)
    x.set_label('diffComm. score', rotation=270, labelpad=15, size=10, weight='normal')

    ax1.axis('off') 

    return G
