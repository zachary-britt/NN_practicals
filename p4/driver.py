import numpy as np
from numpy import setdiff1d as diff
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def shuffle_df(df):
    n = df.count()[0]
    rn = np.random.random(n)
    inds = np.argsort(rn)
    return df.loc[inds]

def tt_split(df, p_test = 0.2):
    n = df.count()[0]
    inds = np.arange(n)
    test_inds = np.random.choice(inds, int(n*p_test), replace=False)
    train_inds = diff(inds, test_inds, assume_unique=True)
    df_test = df.loc[test_inds]
    df_train = df.loc[train_inds]
    return df_train, df_test


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
    df = shuffle_df(df)
    df_train, df_test = tt_split(df, 0.2)
    
