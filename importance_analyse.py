# -*- coding: utf-8 -*-
__author__ = 'Veigar'

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.tree.export import export_graphviz
import pydotplus
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")


def RF_feature_importance(X, y):
    rf = RandomForestRegressor(random_state=161227)
    feat_labels = X.columns[:]
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(20):
        print('%2d) %-*s %f' % (f + 1, 20, feat_labels[indices[f]], importances[indices[f]]))
    plt.figure(figsize=(6, 4))
    plt.barh(range(19, -1, -1), importances[indices][0:20])
    plt.yticks(range(19, -1, -1), feat_labels[indices][0:20])
    plt.ylim([-1, 20])
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig('./pictures/RF Feature Importances.jpg', dpi=400)
    plt.show()


def xgboost_feature_importance(X, y):
    xgb = XGBRegressor(random_state=161227)
    feat_labels = X.columns[:]
    xgb.fit(X, y)
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(20):
        print('%2d) %-*s %f' % (f + 1, 20, feat_labels[indices[f]], importances[indices[f]]))
    plt.figure(figsize=(6, 4))
    plt.barh(range(19, -1, -1), importances[indices][0:20])
    plt.yticks(range(19, -1, -1), feat_labels[indices][0:20])
    plt.ylim([-1, 20])
    plt.title('Feature Importance - XGBoost')
    plt.tight_layout()
    plt.savefig('./pictures/XGB Feature Importances.jpg', dpi=400)
    plt.show()


def ExtraTree_freture_importance(X, y):
    regr = ExtraTreesRegressor(random_state=1227)
    feat_labels = X.columns[:]
    regr.fit(X, y)
    importances = regr.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(20):
        print('%2d) %-*s %f' % (f + 1, 20, feat_labels[indices[f]], importances[indices[f]]))
    plt.figure(figsize=(6, 4))
    plt.barh(range(19, -1, -1), importances[indices][0:20])
    plt.yticks(range(19, -1, -1), feat_labels[indices][0:20])
    plt.ylim([-1, 20])
    plt.title('Feature Importance - Extra Trees')
    plt.tight_layout()
    plt.show()


def Tree_freture_importance_plot(X, y):
    regr = DecisionTreeRegressor()
    feat_labels = X.columns[:]
    regr.fit(X, y)
    out = export_graphviz(regr)
    graph = pydotplus.graph_from_dot_data(out)
    graph.write_pdf('tree.pdf')
