import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ot
import networkx as nx
import itertools
import sklearn
import scanpy as sc

def silhouette_score(phate_op, n_clusters, random_state=None, **kwargs):
    """Compute the Silhouette score on KMeans on the PHATE potential

    Parameters
    ----------
    phate_op : phate.PHATE
        Fitted PHATE operator
    n_clusters : int
        Number of clusters.
    random_state : int or None, optional (default: None)
        Random seed for k-means

    Returns
    -------
    score : float
    """
    cluster_labels = kmeans(
        phate_op, n_clusters=n_clusters, random_state=random_state, **kwargs
    )
    return metrics.silhouette_score(phate_op.diff_potential, cluster_labels)

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
