# %%
import pandas as pd
import numpy as np
import scipy
#import seaborn as sns
#import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#import ot
import networkx as nx
#import itertools
import sklearn
import scanpy as sc
#import sys
#sys.path.append(".")
#from tl import *
from . import tl
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
    complete : flag ; default = 1
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
    Params=coloc_pair_sample (coloc per cell type pair per sample table), exp_condition (non control phenotype to test),
    ctrl_condition (control phenotype), sampleTypes (dataframe with sample names and sample types columns named "sample" and "sampleType")
    """
    pvals=[scipy.stats.ranksums(coloc_pair_sample.loc[coloc_pair_sample.index[sampleTypes.sampleType==exp_condition],c], 
                                        coloc_pair_sample.loc[coloc_pair_sample.index[sampleTypes.sampleType==ctrl_condition],c]).pvalue for c in coloc_pair_sample.columns]
    stat=[scipy.stats.ranksums(coloc_pair_sample.loc[coloc_pair_sample.index[sampleTypes.sampleType==exp_condition],c], 
                                        coloc_pair_sample.loc[coloc_pair_sample.index[sampleTypes.sampleType==ctrl_condition],c]).statistic for c in coloc_pair_sample.columns]


    df=pd.DataFrame([coloc_pair_sample.columns, stat, pvals], index=['pairs', 'statistic', 'p-value']).T
    #myo_iscDF=myo_iscDF.sort_values(['p-value'])
    df.index=df.pairs
    return df

#%%
def spatialNichePlot(adata, cell_types, nicheDF, CTprobs=None, maxCT_col=None, spot_size=1, niche_colors=None, legend_fontsize=7, wspace=0.5, save_name='test.pdf'):
    """ Plot niches and cell types in spatial data (MERFISH / visium slices)
        adata=sample specific spatial anndata object
        CTprobs=sample specific cell type probabilities per spot
        cell_types=categorical series of cell types
        nicheDF=dataframe of cell types and niches (obtained previously via cells_niche_colors())
        niche_colors=series of colors with niche names as indexes
        maxCT_col=cell type column in sc spatial data
    """
    tmp=adata.copy()

    if maxCT_col is None:
        tmp.obs['maxCT']=[CTprobs.columns[np.argmax(CTprobs.loc[idx])] for idx in CTprobs.index]
        tmp.obs.maxCT=tmp.obs.maxCT.astype('category')
        #adata_ex.obs.maxCT.cat.categories=mudata['sc'].obs.cell_subtype2.cat.categories
        for c in np.setdiff1d(cell_types.cat.categories,tmp.obs.maxCT.cat.categories):
            tmp.obs.maxCT = tmp.obs.maxCT.cat.add_categories(c)
        tmp.obs.maxCT=tmp.obs.maxCT.cat.reorder_categories(cell_types.cat.categories)
    else:
        tmp.obs['maxCT']=tmp.obs[maxCT_col]
    
    tmp.obs['niche']='1_'
    for ct in nicheDF.cell:
        tmp.obs.niche[tmp.obs.maxCT==ct]=nicheDF.niche[nicheDF.cell==ct][0]

    tmp.obs.niche=tmp.obs.niche.astype('category')
    for c in np.setdiff1d(nicheDF.niche.cat.categories,tmp.obs.niche.cat.categories):
        tmp.obs.niche = tmp.obs.niche.cat.add_categories(c)
    tmp.obs.niche=tmp.obs.niche.cat.reorder_categories(nicheDF.niche.cat.categories)

    if niche_colors is not None:
        tmp.uns['niche_colors']=niche_colors
    
    #sc.pl.spatial(adata_ex, color=['maxCT', 'niche'], img_key=None, library_id=None, spot_size=200, save='_'+smpl+'_niches_cellST.pdf')
    #with plt.rc_context({"figure.figsize": (2, 2)}):
    sc.pl.spatial(tmp, color=['maxCT', 'niche'], img_key=None, library_id=None, spot_size=spot_size, legend_fontsize=legend_fontsize, wspace=wspace, save=save_name)

#%%

def OvsE_coloc_test(observedColocProbs, expectedColocProbs, cell_types, testDistribution, oneCTinteractions, p=0.05):
    # OvsE ratios
    OvsE=observedColocProbs/expectedColocProbs
    # Log scale
    OvsE_HM=np.log2(OvsE)
    # Filter non significant values
    OvsE_HM[(OvsE_HM>np.quantile(np.log2(testDistribution), p/2)) & (OvsE_HM<np.quantile(np.log2(testDistribution), 1-(p/2)))]=0
    OvsE_HM[oneCTinteractions]=0
    # Reshape into data frame
    OvsE_HMdf=pd.DataFrame(np.array(OvsE_HM).reshape(-1, len(cell_types)))
    OvsE_HMdf.columns=cell_types
    OvsE_HMdf.index=cell_types
    return OvsE_HMdf

#%%
def colocNW(x_diff,adj, cell_group, group_cmap='tab20', ncols=20, clist=None, 
            BTsizedNodes=False, legend_ax=[0.7, 0.05, 0.15, 0.2]):

    """Colocalisation network"""
    ## Just take into account differentially colocalised CT pairs (p<=0.05)
    ## Then scale (exp) to get rid of negative numbers, set missing negative values (which were just very small numbers) to 0 => This is our adjacency matrix

    #cell group cmap
    cmap = plt.cm.get_cmap(group_cmap, ncols)
    if clist == None:
        cgroup_cmap=[mcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    else:
        cgroup_cmap=clist
    
    gCol=nx.from_pandas_adjacency(adj, create_using=nx.Graph)

    ## Edge thickness (NEW)
    for x in list(gCol.edges):
        gCol[x[0]][x[1]]['weight'] = x_diff.loc[x[0], x[1]]

    weights = nx.get_edge_attributes(gCol,'weight').values()
    
    ## Node color groups
    color_group=pd.Series(list(gCol.nodes))
    i=0
    for k in list(cell_group.keys()):
        color_group[[tl.cellCatContained(pair=p, cellCat=cell_group[k]) for p in color_group]]=cgroup_cmap[i]
        i=i+1
    
    ## Edge colors based on diff coloc
    edgeCols=pd.Series(['lightblue' if x_diff.loc[x[0], x[1]]<0 else 'orange' for x in list(gCol.edges)])
    edgeCols.index=[x[0]+'->'+x[1] for x in list(gCol.edges)]
    
    orange_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'orange']
    blue_edges = [(u,v) for u,v in gCol.edges if edgeCols[u+'->'+v] == 'lightblue']

    #normalised scores
    inter=pd.Series(np.abs(pd.Series(list(weights))))
    inter.index=edgeCols.index
    inter[edgeCols=='lightblue']=inter[edgeCols=='lightblue']/np.max(inter[edgeCols=='lightblue'])
    inter[edgeCols=='orange']=inter[edgeCols=='orange']/np.max(inter[edgeCols=='orange'])
    pos = nx.drawing.nx_agraph.graphviz_layout(gCol,prog='neato')

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1]+9)
    
    f,ax1 = plt.subplots(1,1,figsize=(8,8),dpi=100) 

    ## Betweeness statistic sized nodes
    if BTsizedNodes == True:
        ## pagerank sized nodes
        #npg = nx.pagerank(gCol,max_iter=1000, weight=None)
        npg = nx.betweenness_centrality(gCol)
        npg=list(npg.values())
        nx.draw_networkx_nodes(gCol,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)
    else:
        nx.draw_networkx_nodes(gCol,pos,node_color=color_group,ax=ax1)

    nx.draw_networkx_edges(gCol,pos=pos,edge_color=inter[edgeCols=='lightblue'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='lightblue'],ax=ax1, edgelist=blue_edges, edge_cmap=cmap3, edge_vmin=-1, edge_vmax=1)
    nx.draw_networkx_edges(gCol,pos=pos,edge_color=inter[edgeCols=='orange'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='orange'],ax=ax1, edgelist=orange_edges, edge_cmap=cmap4 , edge_vmin=-1, edge_vmax=1)
    nx.draw_networkx_labels(gCol,pos_attrs, font_size=12, font_weight='bold', clip_on=False,ax=ax1)

    sm = plt.cm.ScalarMappable(cmap=cmap4)
    sm._A = []
    sm.set_clim(-1, 1)

    cax = ax1.inset_axes(legend_ax)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.patch.set_alpha(1)
    cax.axis('off')
    x=plt.colorbar(sm, ax=cax, fraction=0.2)
    x.set_label('normalised diffColoc. score', rotation=270, labelpad=15, size=10, weight='normal')
    x.solids.set(alpha=0.3)
    
    return gCol
#%%
