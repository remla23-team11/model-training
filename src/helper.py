import pandas as pd


def get_csv_data(path):
    """
    Read a csv from path with default settings and returns a Dataframe
    """
    return pd.read_csv(path, delimiter='\t', quoting=3)
