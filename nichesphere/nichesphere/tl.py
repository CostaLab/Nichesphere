# %%
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ot
import networkx as nx
import itertools
import sklearn

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
def setPriorDef(sc_adata, ct_col, sample_col, sample, p=1, sampleCTprops=None, ns=None):
    '''
    Calculates the marginal distribution of probabilities of cells to be mapped to a visium slice. Cell type dependent. Cells of the same type 
    get the same prior probability to be mapped. Sums up to 1
    sc_adata=single cell RNA-seq anndata
    ct_col=column in sc_adata.obs indicating cell type
    sample_col=column in sc_adata.obs indicating the sample cells belong to
    sample=name of analyzed sample (function is executed once per sample)
    p=proportion of cells we assume we detect in the scRNA-seq experiment (default=1)
    sampleCTprops=vector of cell type proportions in a sample (we can input this in case we have estimates not coming from the scRNA-seq dataset )
    ns=number of cells from the analysed sample (can be inputed in case our estimate does not come from the scRNA-seq dataset)
    '''
    
    cells_prior=ot.unif(len(sc_adata.obs.index))
    for x in range(len(cells_prior)):
        cells_prior[x]=None
    
    cts=sc_adata.obs[ct_col].value_counts().index
    nk=sc_adata.obs[ct_col].value_counts()
    nks=sc_adata.obs[ct_col].value_counts()
    ns=sc_adata.obs[sample_col].value_counts() if ns is None else ns
    n=len(sc_adata.obs_names)

    for x in range(len(nks)):
        nks[x]=0

    sampleCTprops = sc_adata.obs[sc_adata.obs[sample_col]==sample][ct_col].value_counts()/sc_adata.obs[sc_adata.obs[sample_col]==sample][ct_col].value_counts().sum() if sampleCTprops is None else sampleCTprops
     
    for x in sampleCTprops.index: 
        nks[x]=np.round(sampleCTprops[x]*ns[sample])
    
    for ct in cts:
        cells_prior[list(sc_adata.obs[ct_col]==ct)]=((p*nks[ct]/nk[ct])+(((1-p)*ns[sample])/n))/ns[sample]

        
    
    return cells_prior

# %%

def setPrior_sampleReg(sc_adata, ct_col, sample_col, sample, p=1, sampleCTprops=None, ns=None):
    '''
    Calculates the marginal distribution of probabilities of cells to be mapped to a visium slice. Sample dependent. In cases where a dataset with cells from 
     different slices are present and matching sc/snRNA-seq data is available, cells from the matching slice from the same type 
    get the same prior probability to be mapped based on sample specific cell type proportions. Other cells get a probability based on the total proportion of cells belonging to the slice. 
    Sums up to 1, the probabilities for cells from the mapped visium slice sum up to p
    sc_adata=single cell RNA-seq anndata
    ct_col=column in sc_adata.obs indicating cell type
    sample_col=column in sc_adata.obs indicating the sample cells belong to
    sample=name of analyzed sample (function is executed once per sample)
    p=proportion of cells we assume we detect in the scRNA-seq experiment (default=1)
    sampleCTprops=vector of cell type proportions in a sample (we can input this in case we have estimates not coming from the scRNA-seq dataset )
    ns=number of cells from the analysed sample (can be inputed in case our estimate does not come from the scRNA-seq dataset)
    '''
    cells_prior=ot.unif(len(sc_adata.obs.index))
    for x in range(len(cells_prior)):
        cells_prior[x]=None
    cts=sc_adata.obs[ct_col].value_counts().index
    nk=sc_adata.obs[ct_col].value_counts()
    nks=sc_adata.obs[ct_col].value_counts()
    ns=sc_adata.obs[sample_col].value_counts() if ns is None else ns
    n=len(sc_adata.obs_names)
    for x in range(len(nks)):
        nks[x]=0
    sampleCTprops = sc_adata.obs[sc_adata.obs[sample_col]==sample][ct_col].value_counts()/sc_adata.obs[sc_adata.obs[sample_col]==sample][ct_col].value_counts().sum() if sampleCTprops is None else sampleCTprops
    for x in sampleCTprops.index: 
        nks[x]=np.round(sampleCTprops[x]*ns[sample])
    
 
    for ct in cts:
        cells_prior[list((sc_adata.obs[sample_col]==sample) & (sc_adata.obs[ct_col]==ct))]=p*nks[ct]/(nks[ct]*ns[sample])
        cells_prior[list((sc_adata.obs[sample_col]!=sample) & (sc_adata.obs[ct_col]==ct))]=(1-p)/(n-ns[sample])
    
    return cells_prior
#%%

def get_pairCatDFdir(niches, coloc_probs, coloc_clusts):
    ## niches=niches_dict, coloc_probs=CTcolocalizationP, coloc_clusts=colocClusts
    pairsDir=[]
    for ct in coloc_probs.columns[range(len(coloc_probs.columns)-1)]:
        for ct2 in coloc_probs.columns[range(len(coloc_probs.columns)-1)]:
            pairsDir.append(ct+'->'+ct2)
    pairCatDFdir=pd.DataFrame(pairsDir, columns=['pairs'])
    
    pairCatDFdir['colocCats']=''
    for clust in np.sort(coloc_clusts.unique()):
        pairCatDFdir['colocCats'][[cellCatContained(pair=p, cellCat=coloc_clusts.index[coloc_clusts==clust]) for p in pairCatDFdir.pairs]]=clust+'->'+clust

    for comb in list(itertools.permutations(list(niches.keys()), 2)):
        pairCatDFdir['colocCats'][[(p.split('->')[0] in niches[comb[0]]) & (p.split('->')[1] in niches[comb[1]]) for p in pairCatDFdir.pairs]]=comb[0]+'->'+comb[1]

    return pairCatDFdir
# %%
## w good for PIC and new data
def getColocProbs(CTprobs, spotSamples):
    """ Get colocalisation probabilities (sum across spots of probabilities of each cell type pair being in the same spot) 
    #filesList=list of mapping anndata files 
    #filePrefix=prfix coming before the part of the file name indicating the sample
    CTprobs=cell type probabilities per spot
    spotSamples=sample to which each spot belongs, with cell id as index
    nCellTypes=number of cell types.
    Returns concatenated single sample matrices of celltype x cell type"""
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
    complete=list with repeated values (ct1_x_ct2 and ct2_x_ct1)"""
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
def cellCatContained(pair, cellCat):
    """Boolean element indicating whether the specified element (cell pair) is contained in 
    a list (category)
    pair=evaluated cell type pair
    cellCat=category as a list of cell types"""
    
    contained=[cellType in pair for cellType in cellCat]
    return True in contained
# %%
def cells_niche_colors(CTs, niche_colors, niche_dict):
    niche_df=pd.DataFrame(CTs, columns=['cell'])
    niche_df['niche']=niche_colors.index[0]
    niche_df['color']=niche_colors[0]
    for key in list(niche_dict.keys()):
        niche_df['niche'][[c in niche_dict[key] for c in niche_df.cell]]=key
        niche_df['color'][niche_df['niche']==key]=niche_colors[key]
    niche_df.index=niche_df.cell
    return niche_df


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
    pairCatDF.index=pairCatDF.pairs
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

def processCTKRoutput(ctkrTbl):
    ctkrTbl['gene_A']=ctkrTbl['gene_A'].str.replace('|L', '')
    ctkrTbl['gene_A']=ctkrTbl['gene_A'].str.replace('|R', '')
    ctkrTbl['gene_B']=ctkrTbl['gene_B'].str.replace('|R', '')
    ctkrTbl['gene_B']=ctkrTbl['gene_B'].str.replace('|TF', '')

    ctkrTbl['allpair']=ctkrTbl['allpair'].str.replace('|R', '')
    ctkrTbl['allpair']=ctkrTbl['allpair'].str.replace('|L', '')
    ctkrTbl['allpair']=ctkrTbl['allpair'].str.replace('|TF', '')
    return ctkrTbl



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

def getExpectedColocProbsFromSCs(sc_adata, sample, cell_types, sc_data_sampleCol, sc_adata_annotationCol):
    ## cell_types=cell types list
    ## sc_adata=sc gene expression anndata
    ## sample=name of sample in turn
    ## sc_data_sampleCol=name of the sc_adata.obs column containing sample names to which cells belong
    ## sc_adata_annotationCol=name of the sc_adata.obs column containing cell types of each cell
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

def getAdj_coloc(diffColocDF, pairCatDF, ncells, p=0.05):
    """Colocalisation adjacency matrix (and differential coloc test matrix)"""
    x=pd.DataFrame(pairCatDF.pairs, columns=['pairs'], index=pairCatDF.pairs)
    x['stat']=0

    for i in diffColocDF.pairs:
        if diffColocDF['p-value'][i]<p:
            x.stat[i.split('-')[0]+'->'+i.split('-')[1]]=diffColocDF.statistic[i]
            x.stat[i.split('-')[1]+'->'+i.split('-')[0]]=diffColocDF.statistic[i]
    
    x=pd.Series(x.stat)
    x_diff=pd.DataFrame(np.array(x).reshape(-1, ncells))
    x_diff.columns=unique([x.split('->')[0] for x in pairCatDF.pairs])
    x_diff.index=unique([x.split('->')[0] for x in pairCatDF.pairs])
    

    ##Cosine similarity plus pseudocount
    adj=pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(x_diff)+1)
    adj.index=x_diff.index
    adj.columns=x_diff.columns
    
    ##Cell pairs with not significant differential colocalisation get 0
    adj[x_diff==0]=0
    adj[adj==1]=0
    np.fill_diagonal(adj.values, 0)
    return x_diff,adj

#%%
def getColocFilter(pairCatDF, adj, oneCTints):
    colocFilt=pd.DataFrame(pairCatDF.pairs, columns=['pairs'], 
                       index=pairCatDF.pairs)
    colocFilt['filter']=0

    for i in pairCatDF.pairs:
        colocFilt['filter'][i]=adj.loc[i.split('->')[1],i.split('->')[0]]
    
    colocFilt['filter'][oneCTints]=1
    colocFilt['filter'][colocFilt['filter']>0]=1
    colocFilt=pd.DataFrame(colocFilt['filter'], index=colocFilt.index, columns=['filter'])
    return colocFilt
#%%
def colocNW(x_diff,adj, cell_group, group_cmap='tab20', ncols=20, clist=None, BTsizedNodes=False):

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
        color_group[[cellCatContained(pair=p, cellCat=cell_group[k]) for p in color_group]]=cgroup_cmap[i]
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

    cax = ax1.inset_axes([0.7, 0.05, 0.15, 0.2])
    cax.set_xticks([])
    cax.set_yticks([])
    cax.patch.set_alpha(1)
    cax.axis('off')
    x=plt.colorbar(sm, ax=cax, fraction=0.2)
    x.set_label('normalised diffColoc. score', rotation=270, labelpad=15, size=10, weight='normal')
    x.solids.set(alpha=0.3)
    
    return gCol

#%%
def getAdj_comm(diffCommTbl, pairCatDF, ncells, cat):
    """adjacency matrix and test values for communication (one category at a time)"""
    x=pd.DataFrame(pairCatDF.pairs)
    x['wilcoxStat']=0

    
    for i in diffCommTbl.columns:
        x.wilcoxStat[i]=diffCommTbl[i][cat]

    
    x=pd.Series(x.wilcoxStat)
    x_chem=pd.DataFrame(np.array(x).reshape(-1, ncells))
    x_chem.columns=unique([x.split('->')[0] for x in pairCatDF.pairs])
    x_chem.index=unique([x.split('->')[0] for x in pairCatDF.pairs])

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

def catNW(x_chem,colocNW, cell_group, group_cmap='tab20', ncols=20, color_group=None, plot_title='', clist=None, BTsizedNodes=False):    

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

    cax = ax1.inset_axes([0.7, 0.05, 0.15, 0.2])
    cax.set_xticks([])
    cax.set_yticks([])
    cax.patch.set_alpha(1)
    cax.axis('off')
    x=plt.colorbar(sm, ax=cax, fraction=0.2)
    x.set_label('normalised diffComm. score', rotation=270, labelpad=15, size=10, weight='normal')
    x.solids.set(alpha=0.3)

    ax1.axis('off') 

    return G
#%%
"""My unique func without value reordering"""
def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

#%%
#def PIC_BGdoubletsOEratios(adata_singlets, nmults, annot, singIDs, sep):
def PIC_BGdoubletsOEratios(adata_singlets):
    #nmults = Number of multiplets
    #annot = singlets annotation (character vector)
    #singIDs = singlets IDs (character vector)
    
    ### test diff coloc for PICseq=> 

    ## Generate distribution of O/E ratios for colocalization prob of cell type pairs in randomly generated doublets (background dist)
    
    rdf=pd.DataFrame(adata_singlets.obs.annotation)
    rdf.columns=['annot']
    rdf.index=adata_singlets.obs.index
    rdf['pair']=''
    
    ## Get random singlets pairs
    pairNums=[i for i in range(int(np.round(adata_singlets.obs.shape[0]/2))) for _ in range(2)]
    random.seed(123)
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
