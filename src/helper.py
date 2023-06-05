import pandas as pd


def get_csv_data(path):
    return pd.read_csv(path, delimiter='\t', quoting=3)
