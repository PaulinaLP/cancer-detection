import pandas as pd
import os

def ingest_data (path):
    df_train = pd.read_csv( os.path.join(path,"train-metadata.csv"))
    df_test = pd.read_csv( os.path.join(path,"test-metadata.csv"))
    return df_train, df_test

