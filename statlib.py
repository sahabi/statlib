from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import ceil
import itertools
from scipy.stats.stats import pearsonr
from sklearn import cross_validation
def trasform_data(data):
    for column in data.columns:
        data[column] = preprocessing.LabelEncoder().fit(data[column]).transform(data[column])
    return data

def plot_dist(data):
    columns = list(data.columns)
    f = [x for x in columns]
    ax = [x for x in columns]
    for i,column in enumerate(columns):
        f[i] = plt.figure(i+1)
        ax[i] = f[i].add_subplot(111)
        ax[i].hist(data[column],bins=data[column].unique().max() - data[column].unique().min() + 1 )
        ax[i].set_xticks(np.arange(data[column].unique().min(), data[column].unique().max(), 2))
        ax[i].set_title("P({})".format(column))       
    plt.show()

def plot_cond_dist(data):
    columns = list(data.columns)
    splits = {}
    for column in columns:
        splits[column] = ceil((data[column].unique().max() - data[column].unique().min())/2.)+data[column].unique().min()
    pairs = list(itertools.combinations(columns, 2))
    f = [x for x in range(0,len(pairs))]
    ax = [x for x in range(0,len(pairs)*4)]
    for i,pair in enumerate(pairs):
        f[i] = plt.figure(i+1)
        ax[i] = f[i].add_subplot(221)
        splitted = data[data[pair[1]] >= splits[pair[1]]]
        ax[i].hist(splitted[pair[0]],bins=splitted[pair[0]].unique().max() - splitted[pair[0]].unique().min() + 1 )
        ax[i].set_xticks(np.arange(splitted[pair[0]].unique().min(), splitted[pair[0]].unique().max(), 2))
        ax[i].set_title("P({}|{} >= {})".format(pair[0],pair[1],splits[pair[1]]))

        ax[i+1] = f[i].add_subplot(222)
        splitted = data[data[pair[1]] < splits[pair[1]]]
        ax[i+1].hist(splitted[pair[0]],bins=splitted[pair[0]].unique().max() - splitted[pair[0]].unique().min() + 1 )
        ax[i+1].set_xticks(np.arange(splitted[pair[0]].unique().min(), splitted[pair[0]].unique().max(), 2))
        ax[i+1].set_title("P({}|{} < {})".format(pair[0],pair[1],splits[pair[1]]))

        ax[i+2] = f[i].add_subplot(223)
        splitted = data[data[pair[0]] >= splits[pair[0]]]
        ax[i+2].hist(splitted[pair[1]],bins=splitted[pair[1]].unique().max() - splitted[pair[1]].unique().min() + 1 )
        ax[i+2].set_xticks(np.arange(splitted[pair[1]].unique().min(), splitted[pair[1]].unique().max(), 2))
        ax[i+2].set_title("P({}|{} >= {})".format(pair[1],pair[0],splits[pair[0]]))

        ax[i+3] = f[i].add_subplot(224)
        splitted = data[data[pair[0]] < splits[pair[0]]]
        ax[i+3].hist(splitted[pair[1]],bins=splitted[pair[1]].unique().max() - splitted[pair[1]].unique().min() + 1 )
        ax[i+3].set_xticks(np.arange(splitted[pair[1]].unique().min(), splitted[pair[1]].unique().max(), 2))
        ax[i+3].set_title("P({}|{} < {})".format(pair[1],pair[0],splits[pair[0]]))    
    plt.show()

def t_test(data):
    pairs = list(itertools.combinations(list(data.columns), 2))
    t_stat = {}
    for i,pair in enumerate(pairs):
        t_stat[pair[0]+' and '+pair[1]]= "correlation: {}    p-value: {}".format(pearsonr(data[pair[1]].as_matrix(),data[pair[0]].as_matrix())[0],pearsonr(data[pair[1]].as_matrix(),data[pair[0]].as_matrix())[1])
    return t_stat


def stat_report(data):
    plot_dist(data)
    plot_cond_dist(data)
    return t_test(data)

def prep_tree_tr(data,enc):
    enc.fit(data.as_matrix())
    return enc.transform(data.as_matrix()).toarray()

def prep_tree_ts(data,enc):
    return enc.transform(data.as_matrix()).toarray()

def train(X,Y,model):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                     Y,
                                                                     test_size=0.3,
                                                                     random_state=0)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    #print score
    return (model,score)
