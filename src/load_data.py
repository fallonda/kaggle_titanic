import pandas as pd


def import_data(path):
    df = pd.read_csv(
        path
    )
    return df
