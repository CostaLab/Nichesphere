# %%
import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import glob
import random
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
# Set alpha
cmap1[:,-1] = np.linspace(0, 0.5, cmap.N)
# Create new colormap
cmap1 = ListedColormap(cmap1)

# Choose colormap
cmap = plt.cm.Oranges
# Get the colormap colors
cmap2 = cmap(np.arange(cmap.N))
# Set alpha
cmap2[:,-1] = np.linspace(0, 0.5, cmap.N)
# Create new colormap
cmap2 = ListedColormap(cmap2)



# %%
def locations_score_per_cluster(tissue: novosparc.cm.Tissue, cluster_key: str='clusters') -> ndarray:
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
    arr=[np.array([np.sum(np.array(tissue.gw[:, location][np.argwhere(clusts == cluster).flatten()])) for cluster in clusts_names]) for location in range(len(tissue.locations))]

    
    return pd.DataFrame(arr, columns=clusts_names)

# %%
def setPriorDef(sc_adata, ct_col, sample_col, sample, p=1, sampleCTprops=None, ns=None):
    '''
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
def novosparc_mapping_Def(sc_adata: anndata.AnnData, st_adata: anndata.AnnData, ct_col, cells_prior=None, ref_weight=0.5, thr=0.05, epsilon=5e-4) -> novosparc.cm.Tissue:
    """
    Given two AnnData objects, one single-cell (with cell type column in obs) and one spatial, this function
    will map the expression of the single-cell data onto the spatial data using
    shared highly variable genes as markers.
    :param sc_adata: A spacemake processed single-cell sample.
    :type sc_adata: anndata.AnnData
    :param st_adata: A spacemake processed spatial sample.
    :type st_adata: anndata.AnnData
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
    sc.tl.rank_genes_groups(sc_adata, groupby=ct_col, method='wilcoxon', use_raw=False)
    HVGsDF=pd.DataFrame(0, columns=sc_adata.obs[ct_col].unique(), index=sc_adata.var_names)
    for c in HVGsDF.columns:
        t=pd.Series(sc_adata.uns['rank_genes_groups']['pvals_adj'][c], index=sc_adata.uns['rank_genes_groups']['names'][c])
        HVGsDF[c]=t[sc_adata.var_names]
    thr=thr
    hvgs=(HVGsDF<thr).sum(axis=1).index[[(i!=0) for i in ((HVGsDF<thr).sum(axis=1))]] #genes that are significant for at least one or two groups
    sc_adata_hv = hvgs.to_list()
    
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
    reconst_ad = anndata.AnnData(
        csc_matrix(tissue.sdge.T),
        var = pd.DataFrame(index=tissue.dataset.var_names))
    reconst_ad.X = np.sum(sc_ad.X) * reconst_ad.X / np.sum(reconst_ad.X)
    reconst_ad.obsm['spatial'] = np.array(pd.DataFrame(tissue.locations))
    return(reconst_ad)

# %%
def deconv(sc_ad, st_ad, sc_ct_col, sc_sample_col, p, ref_weight, filename, sample, thr=0.0001, epsilon=5e-4):
    """Whole deconvolution pipeline from cells prior probabilities to be mapped 
    to annData with reconstructed gene expression and cell type proportions as obs"""
    st_ad.obsm['spatial']=st_ad.obsm['X_spatial']
    cells_prior=setPriorDef(sc_ad, ct_col=sc_ct_col, sample_col=sc_sample_col, sample=sample, p=p)
    tissue_reconst = novosparc_mapping_Def(sc_adata = sc_ad, st_adata = st_ad, ct_col=sc_ct_col, cells_prior=cells_prior, ref_weight=ref_weight, thr=thr, epsilon=epsilon)
    reconst_adata=buildReconstAD(tissue_reconst, sc_ad)
    test=locations_score_per_cluster(tissue=tissue_reconst, cluster_key=sc_ct_col)
    t=test.T/test.sum(axis=1)
    #reconst_adata.obs=test/test.sum(axis=1)[0]
    #reconst_adata.obs=test/test.sum(axis=1)
    reconst_adata.obs=t.T
    reconst_adata.write_h5ad(filename+'.h5ad')
    return cells_prior

# %%
## CHANGE W for the future
def getColocProbs(filesList, filePrefix, nCellTypes):
    ## Get colocalisation
    CTcolocalizationP=pd.DataFrame()
    for file in filesList:
    
        sample=file.replace(filePrefix, "")
        sample=sample.replace(".h5ad", "")
    
        testAdata = sc.read(file)
        test=testAdata.obs.iloc[:,0:nCellTypes]
        CTcoloc_P = pd.DataFrame()
        i=0
        for ct in test.columns:
            w=pd.DataFrame([test[ct]*test[col]*len(test.index) for col in test.iloc[:,0:nCellTypes].columns], index=test.iloc[:,0:nCellTypes].columns).sum(axis=1)
            #w=pd.DataFrame([test[ct]*test[col]/len(test.index) for col in test.iloc[:,0:nCellTypes].columns], index=test.iloc[:,0:nCellTypes].columns).sum(axis=1)
            CTcoloc_P = pd.concat([CTcoloc_P, w], axis=1)
            i=i+1
        CTcoloc_P.columns=test.iloc[:,0:nCellTypes].columns
        CTcoloc_P["sample"]=sample
        CTcolocalizationP = pd.concat([CTcolocalizationP, CTcoloc_P])
    
    #CTcolocalizationPnorm.to_csv("./CTcolocalizationProbs_NS_wPrior.csv")
    return CTcolocalizationP

# %%
def reshapeColoc(CTcoloc, oneCTinteractions):   ### matrix of CT pairs x samples
    colocPerSample1=pd.DataFrame()
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
    return colocPerSample1

#%%
def reshapeColoc_complete(CTcoloc):   ### matrix of CT pairs x samples
    colocPerSample1=pd.DataFrame()
    for ct in CTcoloc.columns[range(len(CTcoloc.columns)-1)]:
        x=CTcoloc.iloc[CTcoloc.index==ct,range(len(CTcoloc.columns)-1)]
        for ct2 in CTcoloc.columns[range(len(CTcoloc.columns)-1)]:
            probs=pd.DataFrame(x.iloc[:,x.columns==ct2])
            probs.index=CTcoloc["sample"].unique()
            probs.columns=[ct + '-' + ct2]
            colocPerSample1=pd.concat([colocPerSample1, probs], axis=1)

    #colocPerSample1[np.setdiff1d(colocPerSample1.columns, oneCTinteractions)]=colocPerSample1[np.setdiff1d(colocPerSample1.columns, oneCTinteractions)]*2
    return colocPerSample1

# %%
def cellCatContained(pair, cellCat):
    """Boolean element indicating whether the specified element (cell pair) is contained in 
    a list (category)"""
    #contained=[pair.contains(cellType) for cellType in cellCat]
    contained=[cellType in pair for cellType in cellCat]
    return True in contained

# %%
def calculate_CT_pair_scores(ccommTable, LRscoresCol, CTpairSep):
    """Get cell communication scores per cell type pair by summing all LR scores for that pair.
    If the inversed pair is also available, choose highest value. 
    ccommTable=crossTalkeR results table (single condition)"""
    scores = ccommTable[LRscoresCol].groupby(ccommTable['cellpair']).sum()

    # symmetrization
    for x in scores.index:
        if x.split(CTpairSep)[1]+CTpairSep+x.split(CTpairSep)[0] in scores.index:
            scores[x] = float(np.max([scores[x], scores[x.split(CTpairSep)[1]+CTpairSep+x.split(CTpairSep)[0]]]))

    return scores

def calculate_LR_CT_pair_scores_dir(ccommTable, LRscoresCol, CTpairSep):
    """Get cell communication scores per cell type pair by summing all LR scores for that pair.
    If the inversed pair is also available, choose highest value. 
    ccommTable=crossTalkeR results table (single condition)"""
    ccommTable['allpair']=ccommTable['allpair'].str.replace('\|R', '')
    ccommTable['allpair']=ccommTable['allpair'].str.replace('\|L', '')
    ccommTable['allpair']=ccommTable['allpair'].str.replace('\|TF', '')
    
    scores = ccommTable[LRscoresCol].groupby(ccommTable['allpair']).sum()

    # symmetrization
    #for x in scores.index:
    #    if x.split(CTpairSep)[1]+CTpairSep+x.split(CTpairSep)[0] in scores.index:
    #        scores[x] = float(np.max([scores[x], scores[x.split(CTpairSep)[1]+CTpairSep+x.split(CTpairSep)[0]]]))

    return scores

# %%
#def ctPairScores_perCat_dir(ccommTable, db, dbCatCol, dbMatchCol, ccommMatchCol, ccommLRscoresCol, oneCTinteractions, condition, pairCatDF):
#    db[dbMatchCol]=db[dbMatchCol].str.lower()
#    CTpairScores_byCat=pd.DataFrame()
#    for cat in db[dbCatCol].unique():    
#    
#        ccommScores_plt=calculate_CT_pair_scores_dir(ccommTable=ccommTable[[x in db[dbMatchCol][db[dbCatCol]==cat].tolist() for x in ccommTable[ccommMatchCol].str.lower().tolist()]], LRscoresCol=ccommLRscoresCol, CTpairSep='-')
#        ccommScores_plt=ccommScores_plt[np.setdiff1d(ccommScores_plt.index, oneCTinteractions)]
#    
#        boxplotDF=pairCatDF.loc[ccommScores_plt.index,:]
#        boxplotDF['LRscores']=ccommScores_plt[boxplotDF.index]
#        boxplotDF['LRcat']=cat
#        CTpairScores_byCat=pd.concat([CTpairScores_byCat, boxplotDF])
#    CTpairScores_byCat['condition']=condition
#        
#    return CTpairScores_byCat

def lr_ctPairScores_perCat_dir(ccommTable, db, dbCatCol, dbMatchCol, ccommMatchCol, ccommLRscoresCol, oneCTinteractions, condition, pairCatDF):
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
#### cell pairs by categories scatter plot function

def comm_colocScatter(colocScores, commScores, pairCatDF, catsList, colsList):
    """Colocalisation vs communication scatter plot function"""
    colocScores=colocScores/np.max(colocScores)
    commScores=commScores/np.max(commScores)
    
    order = catsList
    colors = colsList
    
    pairCatDF_Sub=pairCatDF.loc[colocScores.index.intersection(commScores.index),:]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    i=0
    for cat in order:
        if len(commScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index])>0:
            ax1.scatter(colocScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index], commScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index], c=colors[i], label=cat)
            b, a = np.polyfit(colocScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index], commScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index], deg=1)
            # Create sequence of numbers from smallest to highest score 
            xseq = np.linspace(np.min(colocScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index]), np.max(colocScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index]), num=100)
            ax1.plot(xseq, a + b * xseq, c=colors[i])
            ax1.axis('on')
        i=i+1
 
    recs = []
    for i in range(0,len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs,order,loc='lower right', ncol=2, bbox_to_anchor=(2, 0.1))
    plt.xlabel('colocalisation score')
    plt.ylabel('communication score')
    plt.show()

#%%
def comm_colocScatter_stats(colocScores, commScores, pairCatDF, catsList):
    """colocalisation vs communication statistics"""
    colocScores=colocScores/np.max(colocScores)
    commScores=commScores/np.max(commScores)
    order = catsList
    statsDF=pd.DataFrame(index=['overall']+order, columns=['PearsonR', 'pvalue'])
    
    pairCatDF_Sub=pairCatDF.loc[colocScores.index.intersection(commScores.index),:]
    
    statsDF.PearsonR['overall']=scipy.stats.pearsonr(colocScores[colocScores.index.intersection(commScores.index)],commScores[colocScores.index.intersection(commScores.index)]).statistic
    statsDF.pvalue['overall']=scipy.stats.pearsonr(colocScores[colocScores.index.intersection(commScores.index)],commScores[colocScores.index.intersection(commScores.index)]).pvalue
    
    for cat in order:
        if len(commScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index])>2:   
            statsDF.PearsonR[cat]=scipy.stats.pearsonr(colocScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index],commScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index]).statistic
            statsDF.pvalue[cat]=scipy.stats.pearsonr(colocScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index],commScores[pairCatDF_Sub[pairCatDF_Sub.cats==cat].index]).pvalue
            
    return statsDF   

#%%
def makeDiffColocDF(colocProbsC1, colocProbsC2, pairCatDF, catsCol, C1, C2, oneCTinteractions):
    """differential colocalisation DF (probabilities per category and condition)"""
    colocDFc1=pd.DataFrame(colocProbsC1, columns=['colocProb'])
    colocDFc1['cats']=pairCatDF[catsCol][colocDFc1.index]
    colocDFc1['condition']=C1
    colocDFc1=colocDFc1.loc[np.setdiff1d(colocDFc1.index, oneCTinteractions)]
    
    colocDFc2=pd.DataFrame(colocProbsC2, columns=['colocProb'])
    colocDFc2['cats']=pairCatDF[catsCol][colocDFc2.index]
    colocDFc2['condition']=C2
    colocDFc2=colocDFc2.loc[np.setdiff1d(colocDFc2.index, oneCTinteractions)]
    
    colocDF=pd.concat([colocDFc1, colocDFc2])
    colocDF['log10ColocProb']=np.log10(colocDF.colocProb)
    
    return colocDF

#%%
def getDiffColocStats(colocDF, C1, C2):
    """Wilcoxon tests results"""
    diffColocStats=pd.DataFrame(colocDF.cats.unique(), columns=['cats'])
    diffColocStats['wilcoxStat']=[scipy.stats.ranksums(colocDF.colocProb[(colocDF.cats==cat) & (colocDF.condition==C1)], colocDF.colocProb[(colocDF.cats==cat) & (colocDF.condition==C2)]).statistic for cat in colocDF['cats'].unique()]
    diffColocStats['wilcoxPval']=[scipy.stats.ranksums(colocDF.colocProb[(colocDF.cats==cat) & (colocDF.condition==C1)], colocDF.colocProb[(colocDF.cats==cat) & (colocDF.condition==C2)]).pvalue for cat in colocDF['cats'].unique()]
    return diffColocStats

#%%
def commColocStats_perCat(ccommTable, db, dbMatchCol, dbCatCol, ccommMatchCol, ccommLRscoresCol, oneCTinteractions, colocScores, cellCats, pairCatDF, condition):
    db[dbMatchCol]=db[dbMatchCol].str.lower()
    cats_comm_coloc_stats=pd.DataFrame()
    for cat in db[dbCatCol].unique():
        ccommScores_plt=calculate_CT_pair_scores(ccommTable=ccommTable[[x in db[dbMatchCol][db[dbCatCol]==cat].tolist() for x in ccommTable[ccommMatchCol].str.lower().tolist()]], LRscoresCol=ccommLRscoresCol, CTpairSep='-')
        ccommScores_plt=ccommScores_plt[np.setdiff1d(ccommScores_plt.index, oneCTinteractions)]
        colocScores_plt=colocScores
        if len(colocScores_plt.index.intersection(ccommScores_plt.index))>1:
            print(cat)
        
            tmp=comm_colocScatter_stats(colocScores=colocScores, commScores=ccommScores_plt, 
                  pairCatDF=pairCatDF, catsList=cellCats)
            tmp['LRcat']=cat
            cats_comm_coloc_stats=pd.concat([cats_comm_coloc_stats, tmp])
    cats_comm_coloc_stats.index=cats_comm_coloc_stats.index+'_'+cats_comm_coloc_stats.LRcat 
    cats_comm_coloc_stats['condition']=condition
    cats_comm_coloc_stats['cellCat']=[x.split('_')[0] for x in cats_comm_coloc_stats.index]
    cats_comm_coloc_stats['cat']=cats_comm_coloc_stats.LRcat+'_'+cats_comm_coloc_stats.cellCat
    return cats_comm_coloc_stats

#%%
def ctPairScores_perCat(ccommTable, db, dbCatCol, dbMatchCol, ccommMatchCol, ccommLRscoresCol, oneCTinteractions, condition, pairCatDF):
    db[dbMatchCol]=db[dbMatchCol].str.lower()
    CTpairScores_byCat=pd.DataFrame()
    for cat in db[dbCatCol].unique():    
    
        ccommScores_plt=calculate_CT_pair_scores(ccommTable=ccommTable[[x in db[dbMatchCol][db[dbCatCol]==cat].tolist() for x in ccommTable[ccommMatchCol].str.lower().tolist()]], LRscoresCol=ccommLRscoresCol, CTpairSep='->')
        ccommScores_plt=ccommScores_plt[np.setdiff1d(ccommScores_plt.index, oneCTinteractions)]
    
        boxplotDF=pairCatDF.loc[ccommScores_plt.index,:]
        boxplotDF['LRscores']=ccommScores_plt[boxplotDF.index]
        boxplotDF['LRcat']=cat
        CTpairScores_byCat=pd.concat([CTpairScores_byCat, boxplotDF])
    CTpairScores_byCat['condition']=condition
        
    return CTpairScores_byCat

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
def colocNW(x_diff,adj, cell_group, group_cmap='tab20', ncols=20):

    """Colocalisation network"""
    ## Just take into account differentially colocalised CT pairs (p<=0.05)
    ## Then scale (exp) to get rid of negative numbers, set missing negative values (which were just very small numbers) to 0 => This is our adjacency matrix

    #cell group cmap
    cmap = plt.cm.get_cmap(group_cmap, ncols)
    cgroup_cmap=[mcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    
    #x_diff,adj=getAdj_coloc(diffColocDF=diffColocDF, pairCatDF=pairCatDF, ncells=ncells)
    
    gCol=nx.from_pandas_adjacency(adj, create_using=nx.Graph)
    #pos = nx.spring_layout(gCol, k=1.5)
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

    inter=pd.Series(list(weights))/pd.Series(list(weights)).max()
    inter.index=edgeCols.index
    pos = nx.drawing.nx_agraph.graphviz_layout(gCol,prog='neato')
    #for key in colocDists.columns:
    #    pos[key]=hitDist_transform.loc[key].to_numpy()

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.1)

    ## pagerank sized nodes
    npg = nx.pagerank(gCol,max_iter=1000)
    npg=list(npg.values())
    
    ## edge color intensity by weight
    edcol = nx.get_edge_attributes(gCol,'weight')
    edcol = pd.Series(list(edcol.values())/np.max(list(edcol.values())))
    edcol.index=edgeCols.index
    
    f,ax1 = plt.subplots(1,1,figsize=(10,10),dpi=100) 
    nx.draw_networkx_nodes(gCol,pos,node_size=1000*((npg-np.min(npg))/(np.max(npg)-np.min(npg))+1e-5),
        node_color=color_group,ax=ax1)

    nx.draw_networkx_edges(gCol,pos=pos,edge_color=edcol[edgeCols=='lightblue'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='lightblue'],ax=ax1, edgelist=blue_edges, edge_cmap=cmap1)
    nx.draw_networkx_edges(gCol,pos=pos,edge_color=edcol[edgeCols=='orange'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='orange'],ax=ax1, edgelist=orange_edges, edge_cmap=cmap2)
    nx.draw_networkx_labels(gCol,pos,verticalalignment='bottom',
        font_size=12,clip_on=False,ax=ax1)
    #ax1.set_title(f'RepresentativeCCI_{idc}:{nt}')
    ax1.axis('off')
    return gCol

#%%
def getAdj_comm(diffCommTbl, pairCatDF, ncells, cat, p=0.05):
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
## Comm NW per category function
#def catNW(diffCommTbl, cat, pairCatDF, colocNW, ncells, cell_group, group_cmap='tab20', ncols=20):
def catNW(x_chem,adjChem,colocNW, cell_group, group_cmap='tab20', ncols=20, color_group=None):    
    #cell group cmap
    cmap = plt.cm.get_cmap(group_cmap, ncols)
    cgroup_cmap=[mcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
       
    ## Graph of cell type pairs that get closer together in ischemia

    gComm_Chem=nx.from_pandas_adjacency(adjChem, create_using=nx.DiGraph)
    pos = nx.spring_layout(gComm_Chem)

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)

    ## Node color groups
    if color_group is None:

        color_group=pd.Series(list(gComm_Chem.nodes))
        i=0
        for k in list(cell_group.keys()):
            color_group[[cellCatContained(pair=p, cellCat=cell_group[k]) for p in color_group]]=cgroup_cmap[i]
            i=i+1
        
    ## Edge thickness
    weights = nx.get_edge_attributes(gComm_Chem,'weight').values()

    ## Edge colors based on diff coloc
    edgeCols=pd.Series(['lightblue' if x_chem.loc[x[0], x[1]]<0 else 'orange' for x in list(gComm_Chem.edges)])
    edgeCols.index=[x[0]+'->'+x[1] for x in list(gComm_Chem.edges)]
    
    orange_edges = [(u,v) for u,v in gComm_Chem.edges if edgeCols[u+'->'+v] == 'orange']
    blue_edges = [(u,v) for u,v in gComm_Chem.edges if edgeCols[u+'->'+v] == 'lightblue']
    
    inter=pd.Series(list(weights))/pd.Series(list(weights)).max()
    inter.index=edgeCols.index
    pos = nx.drawing.nx_agraph.graphviz_layout(colocNW,prog='neato')
    
    ## pagerank sized nodes
    npg = nx.pagerank(gComm_Chem,max_iter=1000)
    npg=list(npg.values())
    
    f,ax1 = plt.subplots(1,1,figsize=(10,10),dpi=100) 
    nx.draw_networkx_nodes(gComm_Chem,pos,node_size=1000*((npg-np.min(npg))/(np.max(npg)-np.min(npg))+1e-5),
        node_color=color_group,ax=ax1)
    edcol = nx.get_edge_attributes(gComm_Chem,'weight')
    edcol = pd.Series(list(edcol.values())/np.max(list(edcol.values())))
    edcol.index=edgeCols.index
    nx.draw_networkx_edges(gComm_Chem,pos=pos,edge_color=edcol[edgeCols=='lightblue'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='lightblue'],ax=ax1, edgelist=blue_edges, edge_cmap=cmap1)
    nx.draw_networkx_edges(gComm_Chem,pos=pos,edge_color=edcol[edgeCols=='orange'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='orange'],ax=ax1, edgelist=orange_edges, edge_cmap=cmap2)
    nx.draw_networkx_labels(gComm_Chem,pos,verticalalignment='bottom',
        font_size=12,clip_on=False,ax=ax1)
    #ax1.set_title(f'RepresentativeCCI_{idc}:{nt}')
    #plt.title(cat)
    ax1.axis('off')
    return gComm_Chem

#%%

def catNW_test(x_chem,colocNW, cell_group, group_cmap='tab20', ncols=20, color_group=None):    
    colocNW=colocNW.to_directed()
    #cell group cmap
    cmap = plt.cm.get_cmap(group_cmap, ncols)
    cgroup_cmap=[mcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
       
    ## Graph of cell type pairs that get closer together in ischemia

    #gComm_Chem=nx.from_pandas_adjacency(adjChem, create_using=nx.DiGraph)
    #pos = nx.spring_layout(gComm_Chem)
    pos=nx.drawing.nx_agraph.graphviz_layout(colocNW,prog='neato')

    ## Label positions
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)

    ## Node color groups
    if color_group is None:

        color_group=pd.Series(list(colocNW.nodes))
        i=0
        for k in list(cell_group.keys()):
            color_group[[cellCatContained(pair=p, cellCat=cell_group[k]) for p in color_group]]=cgroup_cmap[i]
            i=i+1
        
    ## Edge thickness
    #weights = nx.get_edge_attributes(gComm_Chem,'weight').values()
    weights=[x_chem.loc[x[0], x[1]] for x in list(colocNW.edges)]

    ## Edge colors based on diff coloc
    edgeCols=pd.Series(['lightblue' if x_chem.loc[x[0], x[1]]<0 else 'orange' for x in list(colocNW.edges)])
    edgeCols.index=[x[0]+'->'+x[1] for x in list(colocNW.edges)]
    
    orange_edges = [(u,v) for u,v in colocNW.edges if edgeCols[u+'->'+v] == 'orange']
    blue_edges = [(u,v) for u,v in colocNW.edges if edgeCols[u+'->'+v] == 'lightblue']
    
    inter=pd.Series(list(weights))/pd.Series(list(weights)).max()
    inter.index=edgeCols.index
    pos = nx.drawing.nx_agraph.graphviz_layout(colocNW,prog='neato')
    
    ## pagerank sized nodes
    npg = nx.pagerank(colocNW,max_iter=1000)
    npg=list(npg.values())
    
    f,ax1 = plt.subplots(1,1,figsize=(10,10),dpi=100) 
    nx.draw_networkx_nodes(colocNW,pos,node_size=1000*((npg-np.min(npg))/(np.max(npg)-np.min(npg))+1e-5),
        node_color=color_group,ax=ax1)
    #edcol = nx.get_edge_attributes(gComm_Chem,'weight')
    #edcol = pd.Series(list(edcol.values())/np.max(list(edcol.values())))
    edcol = pd.Series(weights/np.max(weights))
    edcol.index=edgeCols.index
    nx.draw_networkx_edges(colocNW,pos=pos,edge_color=edcol[edgeCols=='lightblue'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='lightblue'],ax=ax1, edgelist=blue_edges, edge_cmap=cmap1)
    nx.draw_networkx_edges(colocNW,pos=pos,edge_color=edcol[edgeCols=='orange'],
        connectionstyle="arc3,rad=0.15",
        width=5*inter[edgeCols=='orange'],ax=ax1, edgelist=orange_edges, edge_cmap=cmap2)
    nx.draw_networkx_labels(colocNW,pos,verticalalignment='bottom',
        font_size=12,clip_on=False,ax=ax1)
    #ax1.set_title(f'RepresentativeCCI_{idc}:{nt}')
    #plt.title(cat)
    ax1.axis('off')
    #return gComm_Chem

    #%%
"""My unique func without value reordering"""
def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]