# %%
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
#import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#import ot
import networkx as nx
#import itertools
import sklearn
#import scanpy as sc
import sys
sys.path.append("./")
from tl import *

from matplotlib.colors import ListedColormap

# Choose colormap
cmap = plt.cm.Blues
# Get the colormap colors
cmap1 = cmap(np.arange(cmap.N))
# Set alpha (transparency)
cmap1[:,-1] = np.linspace(0, 0.5, cmap.N)
# Create new colormap
cmap1 = ListedColormap(cmap1)

# Choose colormap
cmap = plt.cm.Reds
# Get the colormap colors
cmap2 = cmap(np.arange(cmap.N))
# Set alpha
cmap2[:,-1] = np.linspace(0, 0.5, cmap.N)
# Create new colormap
cmap2 = ListedColormap(cmap2)

# Choose colormap
cmap = plt.cm.RdBu
# Get the colormap colors
cmap3 = cmap(np.arange(cmap.N))
# Set alpha (transparency)
cmap3[:,-1] = np.linspace(0, 0.3, cmap.N)
# Create new colormap
cmap3 = ListedColormap(cmap3)

# Choose colormap
cmap = plt.cm.RdBu_r
# Get the colormap colors
cmap4 = cmap(np.arange(cmap.N))
# Set alpha (transparency)
cmap4[:,-1] = np.linspace(0, 0.3, cmap.N)
# Create new colormap
cmap4 = ListedColormap(cmap4)

# %%
def calculate_LR_CT_pair_scores_dir(ccommTable, LRscoresCol):
    """Get cell communication scores per cell type pair per LR pair by summing that LR pair scores for that cell type pair. 
    ccommTable=crossTalkeR results table (single condition)
    LRscoresCol=scores column in the crossTalkeR table
    CTpairSep=pattern separating cell type x from cell type y in pair name"""
    
   
    scores = ccommTable[LRscoresCol].groupby(ccommTable['allpair']).sum()
    return scores
#%%
def lr_ctPairScores_perCat_dir(ccommTable, db, dbCatCol, dbMatchCol, ccommMatchCol, ccommLRscoresCol, oneCTinteractions, condition, pairCatDF):
    """Calculates scores per ligand category from a database"""
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
    '''Makes communication score tables contain the same interactions to be compared'''
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
    """Differential cell communication per LR category"""
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

def plotDiffCcommStatsHM(diffCommTable, min_pval):
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
    plot=sns.clustermap(x_hm, cmap='vlag', center=0)
    plt.setp(plot.ax_heatmap.yaxis.get_majorticklabels(), rotation=1)
    return x_hm, plot

#%%
def getAdj_comm(diffCommTbl, pairCatDF, ncells, cat):
    """adjacency matrix and test values for communication (one category at a time)"""
    x=pd.DataFrame(pairCatDF.cell_pairs)
    x['wilcoxStat']=0

    
    for i in diffCommTbl.columns:
        x.wilcoxStat[i]=diffCommTbl[i][cat]

    
    x=pd.Series(x.wilcoxStat)
    x_chem=pd.DataFrame(np.array(x).reshape(-1, ncells))
    x_chem.columns=unique([x.split('->')[0] for x in pairCatDF.cell_pairs])
    x_chem.index=unique([x.split('->')[0] for x in pairCatDF.cell_pairs])

    ## Another way around: similarities
    ##Cosine similarity
    adjChem=pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(x_chem)+1)
    adjChem.index=x_chem.index
    adjChem.columns=x_chem.columns
    ### 0 similarity for not significant communication
    adjChem[x_chem==0]=0
    adjChem[adjChem==1]=0
    return x_chem,adjChem

#%%

def catNW(x_chem,colocNW, cell_group, group_cmap='tab20', ncols=20, color_group=None, plot_title='', 
          clist=None, BTsizedNodes=False, legend_ax=[0.7, 0.05, 0.15, 0.2]):    

    #cell group cmap
    cmap = plt.cm.get_cmap(group_cmap, ncols)
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
    
    weights=nx.get_edge_attributes(G,'weight').values()

    ## Edge colors based on diff comm
    edgeCols=pd.Series(['lightblue' if x_chem.loc[x[0], x[1]]<0 else 'orange' for x in list(G.edges)])
    edgeCols.index=[x[0]+'->'+x[1] for x in list(G.edges)]
    
    orange_edges = [(u,v) for u,v in G.edges if edgeCols[u+'->'+v] == 'orange']
    blue_edges = [(u,v) for u,v in G.edges if edgeCols[u+'->'+v] == 'lightblue']
    
    inter=pd.Series(np.abs(pd.Series(list(weights))))
    inter.index=edgeCols.index
    inter[edgeCols=='lightblue']=inter[edgeCols=='lightblue']/np.max(inter[edgeCols=='lightblue'])
    inter[edgeCols=='orange']=inter[edgeCols=='orange']/np.max(inter[edgeCols=='orange'])

    pos = nx.drawing.nx_agraph.graphviz_layout(G,prog='neato')

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1]+7)
    
    ###


    to_remove=[(a,b) for a, b, attrs in G.edges(data=True) if attrs["weight"] == 0]
    G.remove_edges_from(to_remove)
    f,ax1 = plt.subplots(1,1,figsize=(8,8),dpi=100) 

    
    if BTsizedNodes == True:
        ## pagerank sized nodes
        #npg = nx.pagerank(gCol,max_iter=1000, weight=None)
        npg = nx.betweenness_centrality(G)
        npg=list(npg.values())

        ## edges positions
        pos_edges=pos        
        
        nx.draw_networkx_nodes(G,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)
        
        
        nx.draw_networkx_edges(G,pos=pos_edges,edge_color=inter[edgeCols=='lightblue'],
            connectionstyle="arc3,rad=0.15", node_size=50+1000*((npg)/(np.max(npg))),
            width=5*inter[edgeCols=='lightblue'],ax=ax1, edgelist=blue_edges, edge_cmap=cmap3,edge_vmin=-1, edge_vmax=1)
        nx.draw_networkx_edges(G,pos=pos_edges,edge_color=inter[edgeCols=='orange'],
            connectionstyle="arc3,rad=0.15", node_size=50+1000*((npg)/(np.max(npg))),
            width=5*inter[edgeCols=='orange'],ax=ax1, edgelist=orange_edges, edge_cmap=cmap4,edge_vmin=-1, edge_vmax=1)

    else:
        pos_edges=pos


        nx.draw_networkx_nodes(G,pos,node_color=color_group,ax=ax1)

        nx.draw_networkx_edges(G,pos=pos_edges,edge_color=inter[edgeCols=='lightblue'],
            connectionstyle="arc3,rad=0.15",
            width=5*inter[edgeCols=='lightblue'],ax=ax1, edgelist=blue_edges, edge_cmap=cmap3,edge_vmin=-1, edge_vmax=1)
        nx.draw_networkx_edges(G,pos=pos_edges,edge_color=inter[edgeCols=='orange'],
            connectionstyle="arc3,rad=0.15",
            width=5*inter[edgeCols=='orange'],ax=ax1, edgelist=orange_edges, edge_cmap=cmap4,edge_vmin=-1, edge_vmax=1)

    
    nx.draw_networkx_labels(G,pos_attrs,verticalalignment='bottom',
        font_size=12,clip_on=False,ax=ax1, font_weight='bold')
    f.suptitle(plot_title)
    
    sm = plt.cm.ScalarMappable(cmap=cmap4)
    sm._A = []
    sm.set_clim(-1, 1)

    cax = ax1.inset_axes(legend_ax)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.patch.set_alpha(1)
    cax.axis('off')
    x=plt.colorbar(sm, ax=cax, fraction=0.2)
    x.set_label('normalised diffComm. score', rotation=270, labelpad=15, size=10, weight='normal')
    x.solids.set(alpha=0.3)

    ax1.axis('off') 

    return G
