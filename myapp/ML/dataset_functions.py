import pandas as pd
import numpy as np
import random as rnd



def getDataFrameValues(url,length,separator):
    data = pd.read_csv(url, engine='python',sep=separator)
    if length > data.shape[0]:
        length = data.shape[0]
    datamatrix = data.to_numpy()
    return datamatrix[:length],data.shape[0]

def getDataFrameColumns(path,separator):
    data = pd.read_csv(path, engine='python',sep=separator)
    #X = data.drop(y_col,axis = 1)
    return data.columns.values.tolist()

def getCategoriesOfColumn(url,column,separator):
    data = pd.read_csv(url, engine='python',sep=separator)
    return data[column].unique().tolist()