import importlib.resources
import pandas as pd

def load_DB():
    """
    Return the NicheSphere biological process LR pair classification database.
    """
    stream = importlib.resources.files("nichesphere") / 'nichesphereDB_pmid.csv'
    return pd.read_csv(stream, index_col=0)