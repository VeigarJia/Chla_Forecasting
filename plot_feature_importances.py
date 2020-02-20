# -*- coding: utf-8 -*-
__author__ = 'Veigar'

import numpy as np
import pandas as pd
from Chla_Forecasting.importance_analyse import *
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import matplotlib

matplotlib.use('TkAgg')
from pylab import *
import warnings

warnings.filterwarnings("ignore")

N0 = pd.read_csv('./2015-2018ALL/N0.csv', header=0).fillna(0)
L1 = pd.read_csv('./2015-2018ALL/L1.csv', header=0).fillna(0)
L2 = pd.read_csv('./2015-2018ALL/L2.csv', header=0).fillna(0)
L3 = pd.read_csv('./2015-2018ALL/L3.csv', header=0).fillna(0)
L4 = pd.read_csv('./2015-2018ALL/L4.csv', header=0).fillna(0)

ALL = N0.join(
    L1.set_index(['Station', 'Year', 'Season']),
    on=['Station', 'Year', 'Season'],
    how='left',
).join(
    L2.set_index(['Station', 'Year', 'Season']),
    on=['Station', 'Year', 'Season'],
    how='left',
).join(
    L3.set_index(['Station', 'Year', 'Season']),
    on=['Station', 'Year', 'Season'],
    how='left',
).join(
    L4.set_index(['Station', 'Year', 'Season']),
    on=['Station', 'Year', 'Season'],
    how='left',
).dropna()

data = ALL.drop(['Chl-a'], axis=1)

classify = pd.concat(
    [data['Station'],
     data['FL'],
     data['FN'],
     data['ST'],
     data['L1_Weather'],
     data['L2_Weather'],
     data['L3_Weather'],
     data['L4_Weather']
     ], axis=1)
regressi = data.drop(['Station'], axis=1) \
    .drop(['FL'], axis=1) \
    .drop(['FN'], axis=1) \
    .drop(['ST'], axis=1) \
    .drop(['L1_Weather'], axis=1) \
    .drop(['L2_Weather'], axis=1) \
    .drop(['L3_Weather'], axis=1) \
    .drop(['L4_Weather'], axis=1)
classify_onehot = pd.get_dummies(classify, sparse=True).astype('int')

X = pd.concat([classify_onehot, regressi], axis=1)
y = ALL['Chl-a']

# RF_feature_importance(X, y)
xgboost_feature_importance(X, y)
