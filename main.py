from statlib import trasform_data, stat_report, prep_tree_tr, train
import itertools
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, cross_validation
from sklearn.tree import DecisionTreeRegressor
from math import ceil
from scipy.stats import ttest_ind
data = '~/Documents/dataprojects/data/student-mat.csv'
data2 = '~/Documents/dataprojects/data/student-por.csv'
students_df2 = pd.read_csv(data2,sep=';')
students_df = pd.read_csv(data,sep=';')
students_df = trasform_data(students_df)    
enc = preprocessing.OneHotEncoder()
X = students_df.drop(['Dalc','Walc'],1)
X = prep_tree_tr(X,enc)
students_df['alch'] = students_df.Walc*2 + students_df.Dalc*5
Y = students_df['alch'].as_matrix()
tree = DecisionTreeRegressor()
(tree,score) = train(X,Y,tree)
print stat_report(students_df[['sex','age','alch']])
print score
print tree