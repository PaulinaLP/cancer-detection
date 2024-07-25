import pandas as pd
import os

df_train = pd.read_csv( "train-metadata.csv")

for i in range (10000,40000,10000):
    batch = df_train.iloc[i:(i+10000),:]
    batch.to_csv("input/batch"+str(int(i/10000))+".csv")
   