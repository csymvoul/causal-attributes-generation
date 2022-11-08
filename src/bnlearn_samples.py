import bnlearn as bn
import numpy as np
import pandas as pd

def bnlearn_samples():
    # Load the data
    file = open("user_info.csv", 'r')
    data = np.loadtxt(file, delimiter=",", skiprows=1)
    columns=['USER_ID','CURRENT_USER_NODE_ID',
             'CURRENT_DATA_NODE_ID','DISTANCE','USER_TYPE']
    df_raw = pd.DataFrame(data)
    df_raw.columns = columns
    dfhot, dfnum = bn.df2onehot(df_raw)

    model = bn.structure_learning.fit(dfnum)
    # model['adjmat']
    print(model['adjmat'])

bnlearn_samples()