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
