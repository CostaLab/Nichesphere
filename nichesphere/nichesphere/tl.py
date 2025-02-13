# %%
import pandas as pd
import numpy as np
#import scipy
#import seaborn as sns
import random
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#import ot
#import networkx as nx
import itertools
#import sklearn
#import scanpy as sc

#from matplotlib.colors import ListedColormap


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
"""My unique func without value reordering"""
def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

#%%
def pval_filtered_HMdf(testDF, oneCTinteractions, p, cell_types):
    """ Get dataframe for scores heatmap from dataframe obtained from diffColoc_test()
    
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

def get_pairCatDFdir(niches, coloc_probs, coloc_clusts):
    
    """Get dataframe of cell pair to niche pair correspondence

    Parameters
    ----------
    niches : dictionary
        Dictionary with niche names as keys and their corresponding cell types as items (list per niche)
    coloc_probs : pd.DataFrame
        Concatenated dataframes of cell type pairs co-localization probabilities per sample (or any dataframe with cell types as columns)
    coloc_clusts : pd.Series
        Series of niches / clusters with cell types as index

    Returns
    -------
    pairCatDFdir : pd.DataFrame
        dataframe of cell pairs and corresponding niche pairs
    """
    pairsDir=[]
    for ct in coloc_probs.columns[range(len(coloc_probs.columns)-1)]:
        for ct2 in coloc_probs.columns[range(len(coloc_probs.columns)-1)]:
            pairsDir.append(ct+'->'+ct2)
    pairCatDFdir=pd.DataFrame(pairsDir, columns=['cell_pairs'])
    
    pairCatDFdir['niche_pairs']=''
    for clust in np.sort(coloc_clusts.unique()):
        pairCatDFdir['niche_pairs'][[cellCatContained(pair=p, cellCat=coloc_clusts.index[coloc_clusts==clust]) for p in pairCatDFdir.cell_pairs]]=clust+'->'+clust

    for comb in list(itertools.permutations(list(niches.keys()), 2)):
        pairCatDFdir['niche_pairs'][[(p.split('->')[0] in niches[comb[0]]) & (p.split('->')[1] in niches[comb[1]]) for p in pairCatDFdir.cell_pairs]]=comb[0]+'->'+comb[1]

    return pairCatDFdir
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
def getColocFilter(pairCatDF, adj, oneCTints):
    colocFilt=pd.DataFrame(pairCatDF.cell_pairs, columns=['cell_pairs'], 
                       index=pairCatDF.cell_pairs)
    colocFilt['filter']=0

    for i in pairCatDF.cell_pairs:
        colocFilt['filter'][i]=adj.loc[i.split('->')[1],i.split('->')[0]]
    
    colocFilt['filter'][oneCTints]=1
    colocFilt['filter'][colocFilt['filter']>0]=1
    colocFilt=pd.DataFrame(colocFilt['filter'], index=colocFilt.index, columns=['filter'])
    return colocFilt
