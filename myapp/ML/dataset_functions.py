import pandas as pd
import numpy as np
import random as rnd



def getDataFrameValues(url,length):
    data = pd.read_csv(url, engine='python')
    if length > data.shape[0]:
        length = data.shape[0]
    datamatrix = data.to_numpy()
    return datamatrix[:length],data.shape[0]

def getDataFrameColumns(path):
    data = pd.read_csv(path, engine='python')
    #X = data.drop(y_col,axis = 1)
    return data.columns.values.tolist()

def getCategoriesOfColumn(url,column):
    data = pd.read_csv(url, engine='python')
    return data[column].unique().tolist()