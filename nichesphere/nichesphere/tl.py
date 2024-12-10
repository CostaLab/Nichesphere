# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import glob
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import ot
import networkx as nx
import statsmodels
import logging
import novosparc
import anndata
import itertools
import sklearn

from scipy.sparse import csc_matrix

from numpy import ndarray

from statsmodels import stats

from matplotlib.colors import ListedColormap

logger_name = 'spacemake.spatial.novosparc_integration'
logger = logging.getLogger(logger_name)

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
def locations_score_per_cluster(tissue: novosparc.cm.Tissue, cluster_key: str='clusters', sc_mapping_file='sc_mapping.csv') -> ndarray:
    """Maps the annotated clusters obtained from the scRNA-seq analysis onto
    the tissue space.
    :param: tissue:    the novosparc tissue object containing the gene expression data,
                      the clusters annotation and the spatial reconstruction. Assumes
                      that the cluster annotation exists in the underlying anndata object.
    :type: tissue: novosparc.cm.Tissue
    :param: cluster_key: the key of the clustering to be used for deduction
    :type: cluster_key: str
    :returns: A numpy list of cluster names, one per each spot
    :rtype: numpy.ndarray
    """

    clusts = tissue.dataset.obs[cluster_key].to_numpy().flatten()
    clusts_names = np.unique(clusts)


    #arr=[np.array([np.median(np.array(tissue.gw[:, location][np.argwhere(clusts == cluster).flatten()])) for cluster in clusts_names]) for location in range(len(tissue.locations))]
    t=pd.DataFrame(tissue.gw)
    t.to_csv(sc_mapping_file)
    arr=[np.array([np.sum(np.array(tissue.gw[:, location][np.argwhere(clusts == cluster).flatten()])) for cluster in clusts_names]) for location in range(len(tissue.locations))]

    
    return pd.DataFrame(arr, columns=clusts_names)

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
    
    #for ct in cts:
    #    cells_prior[list(sc_adata.obs[ct_col]==ct)]=((p*nks[ct]/nk[ct])+(((1-p)*ns[sample])/n))/ns[sample]

    #for ct in cts:
    #    cells_prior[list((sc_adata.obs[sample_col]==sample) & (sc_adata.obs[ct_col]==ct))]=(nks[ct]/nk[ct])/(ns[sample]*ns[sample]/n)
    #    cells_prior[list((sc_adata.obs[sample_col]!=sample) & (sc_adata.obs[ct_col]==ct))]=(ns[sample]/n)/(ns[sample]*(n-ns[sample])/n)
 
    for ct in cts:
        cells_prior[list((sc_adata.obs[sample_col]==sample) & (sc_adata.obs[ct_col]==ct))]=p*nks[ct]/(nks[ct]*ns[sample])
        #cells_prior[list((sc_adata.obs[sample_col]!=sample) & (sc_adata.obs[ct_col]==ct))]=(1-p)*(ns[sample]/n)/((n-ns[sample])*ns[sample]/n)
        cells_prior[list((sc_adata.obs[sample_col]!=sample) & (sc_adata.obs[ct_col]==ct))]=(1-p)/(n-ns[sample])
    
    return cells_prior
#%%

def novosparc_mapping_Def(sc_adata: anndata.AnnData, st_adata: anndata.AnnData, ct_col, cells_prior=None, ref_weight=0.5, thr=0.05, epsilon=5e-4) -> novosparc.cm.Tissue:
    """
    Given two AnnData objects, one single-cell (with cell type column in obs) and one spatial, this function
    will map the expression of the single-cell data onto the spatial data using
    shared highly variable genes as markers.
    :param sc_adata: A spacemake processed single-cell sample.
    :type sc_adata: anndata.AnnData
    :param st_adata: A spacemake processed spatial sample.
    :type st_adata: anndata.AnnData
    ct_col=cell type column in scRNA-seq anndata obs
    cells_prior=marginal probabilities for cell to be mapped
    ref_weight=how much the visium gene expression data will be taken into account 
    thr=minimum p-value to take highly variable genes from the scRNA-seq dataset
    epsilon=enthropy regularisation parameter 
    :returns: A novosparc.cm.Tissue object with 2D expression information.
        The locations of the Tissue will be identical to the locations of 
        the spatial sample.
    :rtype: novosparc.cm.Tissue
    """

    from scanpy._utils import check_nonnegative_integers
    from scipy.sparse import csc_matrix

    logger.info('Mapping single-cell data onto spatial data with novosparc')

    if (check_nonnegative_integers(sc_adata.X)
        or check_nonnegative_integers(st_adata.X)
    ):
        # if any of the inputs is count-data, raise error
        raise ValueError(f'External dge seems to contain raw counts. '+
            'Normalised values are expected for both sc_adata and st_adata.')

    # calculate variable genes for both
    sc.tl.rank_genes_groups(sc_adata, groupby=ct_col, method='wilcoxon', use_raw=False, copy=False)
    HVGsDF=pd.DataFrame(0, columns=sc_adata.obs[ct_col].unique(), index=sc_adata.var_names)
    for c in HVGsDF.columns:
        t=pd.Series(sc_adata.uns['rank_genes_groups']['pvals_adj'][c], index=sc_adata.uns['rank_genes_groups']['names'][c])
        HVGsDF[c]=t[sc_adata.var_names]
    thr=thr
    hvgs=(HVGsDF<thr).sum(axis=1).index[[(i!=0) for i in ((HVGsDF<thr).sum(axis=1))]] #genes that are significant for at least one or two groups
    sc_adata_hv = hvgs.to_list()
    
    st_adata.var_names_make_unique()
    sc.pp.highly_variable_genes(st_adata)
    st_adata_hv = st_adata.var_names[st_adata.var.highly_variable].to_list()

    markers = list(set(sc_adata_hv).intersection(st_adata_hv))

    logger.info(f'{len(markers)} number of common markers found. Using them' +
        ' for reconstruction')


    if not 'spatial' in st_adata.obsm:
        raise TypeError(f'The object provided to st_adata is not spatial')


    # make dense dataset
    if(scipy.sparse.issparse(sc_adata.X)):
        dense_dataset = anndata.AnnData(
            sc_adata.X.toarray(),
            obs = sc_adata.obs,
            var = sc_adata.var)
    else:
        dense_dataset = anndata.AnnData(
            sc_adata.X,
            obs = sc_adata.obs,
            var = sc_adata.var)
    
    marker_ix = [dense_dataset.var.index.get_loc(marker) for marker in markers]

    tissue = novosparc.cm.Tissue(dataset=dense_dataset, locations=st_adata.obsm['spatial'])
    num_neighbors_s = 5
    num_neighbors_t = 5

    tissue.setup_linear_cost(markers_to_use=marker_ix, atlas_matrix=st_adata.to_df()[markers].values,
                             markers_metric='minkowski', markers_metric_p=2)
    tissue.setup_smooth_costs(dge_rep = sc_adata.to_df()[sc_adata_hv],
                              num_neighbors_s=num_neighbors_s,
                              num_neighbors_t=num_neighbors_t)

    tissue.reconstruct(alpha_linear=ref_weight, epsilon=epsilon, p_expression=cells_prior)

    return tissue

# %%
def buildReconstAD(tissue, sc_ad):
    """Makes anndata object from novosparc tissue object and reconstructs gene expression based on cell type proportions and single cell
    expression data
    tissue=novosparc tissue object
    sc_ad=scRNA-seq anndata object"""
    reconst_ad = anndata.AnnData(
        csc_matrix(tissue.sdge.T),
        var = pd.DataFrame(index=tissue.dataset.var_names))
    reconst_ad.X = np.sum(sc_ad.X) * reconst_ad.X / np.sum(reconst_ad.X)
    reconst_ad.obsm['spatial'] = np.array(pd.DataFrame(tissue.locations))
    return(reconst_ad)

# %%
def deconv(sc_ad, st_ad, sc_ct_col, sc_sample_col, p, ref_weight, filename, sample, thr=0.0001, epsilon=5e-4, sc_mapping_file='sc_mapping.csv'):
    """Whole deconvolution pipeline from cells prior probabilities to be mapped 
    to annData with reconstructed gene expression and cell type proportions as obs"""
    st_ad.uns['log1p']["base"]=None
    if 'X_spatial' in st_ad.obsm:
        st_ad.obsm['spatial']=st_ad.obsm['X_spatial']
    cells_prior=setPriorDef(sc_ad, ct_col=sc_ct_col, sample_col=sc_sample_col, sample=sample, p=p)
    tissue_reconst = novosparc_mapping_Def(sc_adata = sc_ad, st_adata = st_ad, ct_col=sc_ct_col, cells_prior=cells_prior, ref_weight=ref_weight, thr=thr, epsilon=epsilon)
    reconst_adata=buildReconstAD(tissue_reconst, sc_ad)
    test=locations_score_per_cluster(tissue=tissue_reconst, cluster_key=sc_ct_col, sc_mapping_file=sc_mapping_file)
    t=test.T/test.sum(axis=1)
    #reconst_adata.obs=test/test.sum(axis=1)[0]
    #reconst_adata.obs=test/test.sum(axis=1)
    reconst_adata.obs=t.T
    reconst_adata.write_h5ad(filename+'.h5ad')
    return cells_prior
#%%
def deconv_sc(sc_ad, st_ad, sc_ct_col, sc_sample_col, p, ref_weight, filename, sample, thr=0.0001, epsilon=5e-4, sc_mapping_file='sc_mapping.csv'):
    """Whole deconvolution pipeline from cells prior probabilities to be mapped 
    to annData with reconstructed gene expression and cell type proportions as obs"""
    st_ad.uns['log1p']["base"]=None
    if 'X_spatial' in st_ad.obsm:
        st_ad.obsm['spatial']=st_ad.obsm['X_spatial']
    cells_prior=setPrior_sampleReg(sc_ad, ct_col=sc_ct_col, sample_col=sc_sample_col, sample=sample, p=p)
    tissue_reconst = novosparc_mapping_Def(sc_adata = sc_ad, st_adata = st_ad, ct_col=sc_ct_col, cells_prior=cells_prior, ref_weight=ref_weight, thr=thr, epsilon=epsilon)
    reconst_adata=buildReconstAD(tissue_reconst, sc_ad)
    test=locations_score_per_cluster(tissue=tissue_reconst, cluster_key=sc_ct_col, sc_mapping_file=sc_mapping_file)
    t=test.T/test.sum(axis=1)
    #reconst_adata.obs=test/test.sum(axis=1)[0]
    #reconst_adata.obs=test/test.sum(axis=1)
    reconst_adata.obs=t.T
    reconst_adata.write_h5ad(filename+'.h5ad')
    return cells_prior

# %%
## w good for PIC and new data
def getColocProbs(filesList, filePrefix, nCellTypes):
    """ Get colocalisation probabilities (sum across spots of probabilities of each cell type pair being in the same spot) 
    filesList=list of mapping anndata files 
    filePrefix=prfix coming before the part of the file name indicating the sample
    nCellTypes=number of cell types.
    Returns concatenated single sample matrices of celltype x cell type"""
    CTcolocalizationP=pd.DataFrame()
    for file in filesList:
    
        sample=file.replace(filePrefix, "")
        sample=sample.replace(".h5ad", "")
    
        testAdata = sc.read(file)
        test=testAdata.obs.iloc[:,0:nCellTypes]
        CTcoloc_P = pd.DataFrame()
        i=0
        for ct in test.columns:
            #w=pd.DataFrame([test[ct]*test[col]*len(test.index) for col in test.iloc[:,0:nCellTypes].columns], index=test.iloc[:,0:nCellTypes].columns).sum(axis=1)
            w=pd.DataFrame([test[ct]*test[col]/len(test.index) for col in test.iloc[:,0:nCellTypes].columns], index=test.iloc[:,0:nCellTypes].columns).sum(axis=1)
            CTcoloc_P = pd.concat([CTcoloc_P, w], axis=1)
            i=i+1
        CTcoloc_P.columns=test.iloc[:,0:nCellTypes].columns
        CTcoloc_P["sample"]=sample
        CTcolocalizationP = pd.concat([CTcolocalizationP, CTcoloc_P])
    
    #CTcolocalizationPnorm.to_csv("./CTcolocalizationProbs_NS_wPrior.csv")
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
    #contained=[pair.contains(cellType) for cellType in cellCat]
    contained=[cellType in pair for cellType in cellCat]
    return True in contained
# %%
def calculate_LR_CT_pair_scores_dir(ccommTable, LRscoresCol, CTpairSep):
    """Get cell communication scores per cell type pair per LR pair by summing that LR pair scores for that cell type pair. 
    ccommTable=crossTalkeR results table (single condition)
    LRscoresCol=scores column in the crossTalkeR table
    CTpairSep=pattern separating cell type x from cell type y in pair name"""
    
    #ccommTable['allpair']=ccommTable['allpair'].str.replace('|R', '')
    #ccommTable['allpair']=ccommTable['allpair'].str.replace('|L', '')
    #ccommTable['allpair']=ccommTable['allpair'].str.replace('|TF', '')
   
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
    
        ccommScores_plt=pd.DataFrame(calculate_LR_CT_pair_scores_dir(ccommTable=ccommTable[[x in db[dbMatchCol][db[dbCatCol]==cat].tolist() for x in ccommTable[ccommMatchCol].str.lower().tolist()]], LRscoresCol=ccommLRscoresCol, CTpairSep='@'))
        #ccommScores_plt=ccommScores_plt[np.setdiff1d(ccommScores_plt.index, oneCTinteractions)]
        
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
    sns.set(font_scale=1.5)
    plot=sns.clustermap(x_hm, cmap='vlag', center=0)
    #plot.set_yticklabels(rotation=90)
    plt.setp(plot.ax_heatmap.yaxis.get_majorticklabels(), rotation=1)
    #plt.show()
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
    
    
    #x_diff=x_diff.loc[x_diff.index!='prolif',x_diff.columns!='prolif']

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
    
    #x_diff,adj=getAdj_coloc(diffColocDF=diffColocDF, pairCatDF=pairCatDF, ncells=ncells)
    
    gCol=nx.from_pandas_adjacency(adj, create_using=nx.Graph)
    #pos = nx.spring_layout(gCol, k=1.5)

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
    #inter=np.abs(pd.Series(list(weights))/pd.Series(list(weights)).max())
    inter=pd.Series(np.abs(pd.Series(list(weights))))
    inter.index=edgeCols.index
    inter[edgeCols=='lightblue']=inter[edgeCols=='lightblue']/np.max(inter[edgeCols=='lightblue'])
    inter[edgeCols=='orange']=inter[edgeCols=='orange']/np.max(inter[edgeCols=='orange'])
    pos = nx.drawing.nx_agraph.graphviz_layout(gCol,prog='neato')
    #for key in colocDists.columns:
    #    pos[key]=hitDist_transform.loc[key].to_numpy()

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1]+9)
    
    ## edge color intensity by weight
    #edcol = nx.get_edge_attributes(gCol,'weight')
    #edcol = pd.Series(list(edcol.values())/np.max(list(edcol.values())))
    #edcol.index=edgeCols.index
    
    f,ax1 = plt.subplots(1,1,figsize=(8,8),dpi=100) 

    ## Betweeness statistic sized nodes
    if BTsizedNodes == True:
        ## pagerank sized nodes
        #npg = nx.pagerank(gCol,max_iter=1000, weight=None)
        npg = nx.betweenness_centrality(gCol)
        npg=list(npg.values())
        #nx.draw_networkx_nodes(gCol,pos,node_size=1000*((npg-np.min(npg))/(np.max(npg)-np.min(npg))+1e-5),
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
    #nx.draw_networkx_labels(gCol,pos_attrs,verticalalignment='bottom',
    #    font_size=12,clip_on=False,ax=ax1)
    
    #sm = plt.cm.ScalarMappable(cmap=cmap4, norm=plt.Normalize(vmin = -1, vmax=1))
    #sm = plt.cm.ScalarMappable(cmap=cmap4)
    #sm._A = []
    #plt.colorbar(sm, ax=ax1)

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
    #plt.axis('off')
    
    #ax1.set_title(f'RepresentativeCCI_{idc}:{nt}')
    #ax1.savefig('figures/colocNW.pdf')
    #ax1.axis('off')
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
    
    #inter=pd.Series(list(weights))/pd.Series(list(weights)).max()
    #inter.index=edgeCols.index

    inter=pd.Series(np.abs(pd.Series(list(weights))))
    inter.index=edgeCols.index
    inter[edgeCols=='lightblue']=inter[edgeCols=='lightblue']/np.max(inter[edgeCols=='lightblue'])
    inter[edgeCols=='orange']=inter[edgeCols=='orange']/np.max(inter[edgeCols=='orange'])

    pos = nx.drawing.nx_agraph.graphviz_layout(G,prog='neato')

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1]+7)
    
    ## edge color intensity by weight 
    
    #edcol = nx.get_edge_attributes(G,'weight')
    #edcol = pd.Series(list(edcol.values())/np.max(list(edcol.values())))
    #edcol.index=edgeCols.index
    
    ###

    to_remove=[(a,b) for a, b, attrs in G.edges(data=True) if attrs["weight"] == 0]
    G.remove_edges_from(to_remove)
    f,ax1 = plt.subplots(1,1,figsize=(8,8),dpi=100) 
    #if PGsizedNodes == True:
    #    ## pagerank sized nodes
    #    npg = nx.pagerank(G,max_iter=1000, weight=None)
    #    npg=list(npg.values())
    #    #nx.draw_networkx_nodes(G,pos,node_size=1000*((npg-np.min(npg))/(np.max(npg)-np.min(npg))+1e-5),
    #    nx.draw_networkx_nodes(G,pos,node_size=1000*((npg)/(np.max(npg))),
    #        node_color=color_group,ax=ax1)
    #else:
    #    nx.draw_networkx_nodes(G,pos, node_color=color_group,ax=ax1)

    if BTsizedNodes == True:
        ## pagerank sized nodes
        #npg = nx.pagerank(gCol,max_iter=1000, weight=None)
        npg = nx.betweenness_centrality(G)
        npg=list(npg.values())
        
        nx.draw_networkx_nodes(G,pos,node_size=50+1000*((npg)/(np.max(npg))),
            node_color=color_group,ax=ax1)
    else:
        nx.draw_networkx_nodes(G,pos,node_color=color_group,ax=ax1)
    #nx.draw_networkx_edges(G,pos=pos,edge_color=edcol[edgeCols=='lightblue'],
    #    connectionstyle="arc3,rad=0.15",
    #    width=5*inter[edgeCols=='lightblue'],ax=ax1, edgelist=blue_edges, edge_cmap=cmap1)
    #nx.draw_networkx_edges(G,pos=pos,edge_color=edcol[edgeCols=='orange'],
    #    connectionstyle="arc3,rad=0.15",
    #    width=5*inter[edgeCols=='orange'],ax=ax1, edgelist=orange_edges, edge_cmap=cmap2)
    nx.draw_networkx_edges(G,pos=pos,edge_color=inter[edgeCols=='lightblue'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='lightblue'],ax=ax1, edgelist=blue_edges, edge_cmap=cmap3,edge_vmin=-1, edge_vmax=1)
    nx.draw_networkx_edges(G,pos=pos,edge_color=inter[edgeCols=='orange'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='orange'],ax=ax1, edgelist=orange_edges, edge_cmap=cmap4,edge_vmin=-1, edge_vmax=1)
    nx.draw_networkx_labels(G,pos_attrs,verticalalignment='bottom',
        font_size=12,clip_on=False,ax=ax1, font_weight='bold')
    f.suptitle(plot_title)
    #ax2.set_facecolor('white')
    #plt.show()
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
def PIC_BGdoubletsOEratios(adata_singlets, nmults, annot, singIDs, sep):
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
    #pairProbs=[]
    for x in probs:
        pairProbs=pd.concat([pairProbs, pd.DataFrame(x*probs)])
        #pairProbs.append(x*probs)
    pci=[]
    for x in probs.index:
        pci.append((x+'-'+probs.index.astype(str)).tolist())    
    pairProbs.index=[item for sublist in pci for item in sublist]

    ## Observed probabilities of cell type pairs
    pairProbsO=pd.Series(pairCounts).value_counts()/pd.Series(pairCounts).value_counts().sum()
    ## O/E ratios
    OEratios=pairProbsO/pairProbs['count'][pairProbsO.index]
    return OEratios
    #return pairProbsO, pairProbs