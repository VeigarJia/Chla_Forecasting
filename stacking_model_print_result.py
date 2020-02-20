# -*- coding: utf-8 -*-
__author__ = 'Veigar'
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import *
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np
import warnings

warnings.filterwarnings("ignore")

origin = pd.read_csv('Feature_Selection.csv')
# This file is generated from pre_processing.py
origin = origin.drop(['Station'], axis=1) \
    .drop(['Function'], axis=1) \
    .drop(['Function Name'], axis=1) \
    .drop(['Nearest Sewage Type'], axis=1) \
    .drop(['L1_Weather'], axis=1) \
    .drop(['L2_Weather'], axis=1) \
    .drop(['L3_Weather'], axis=1) \
    .drop(['L4_Weather'], axis=1)
others = pd.read_csv('Cluster_Results.csv')
# This file is generated from cluster_function.py
all_features = origin.join(
    others.set_index(['x', 'y']),
    on=['x', 'y'],
    how='left',
)

method = 'KMeans'  # 'AgglomerativeClustering','KMeans','MiniBatchKMeans','GM'


class regr_models:
    # KNN
    model_KNN = KNeighborsRegressor()
    param_grid_KNN = dict(p=[1],
                          n_neighbors=[2],
                          )
    regr_KNN = GridSearchCV(model_KNN, param_grid_KNN, scoring='r2')
    # XGB
    model_XGB = XGBRegressor()
    param_grid_XGB = dict(
        learning_rate=[0.2, 0.3],
        max_depth=[4],
        n_estimators=[10, 15, 20],
        seed=[161227])
    regr_XGB = GridSearchCV(model_XGB, param_grid_XGB, verbose=False, scoring='r2')

    # RF
    model_RF = RandomForestRegressor()
    param_grid_RF = dict(
        n_estimators=[10],
        max_depth=[5, 10],
        max_features=["sqrt"],
        random_state=[161227]
    )
    regr_RF = GridSearchCV(model_RF,
                           param_grid_RF,
                           cv=KFold(n_splits=5, random_state=161227),
                           scoring='r2')

    # DT
    model_DT = DecisionTreeRegressor()
    param_grid_DT = dict(random_state=[161227],
                         max_features=["sqrt", "log2"],
                         max_depth=np.arange(2, 6, 1),
                         splitter=["best", "random"]
                         )
    regr_DT = GridSearchCV(model_DT, param_grid_DT, scoring='r2')

    # SVR
    model_SVR = SVR()
    # 把Chla取消归一化后效果很差
    param_grid_SVR = dict(C=[5, 10, 20],
                          epsilon=[0.10],
                          kernel=['linear', 'rbf'],
                          gamma=['scale']
                          )
    regr_SVR = GridSearchCV(model_SVR,
                            param_grid_SVR,
                            scoring='r2')

    # MLP
    model_MLP = MLPRegressor()
    param_grid_MLP = dict(learning_rate_init=[0.01, 0.02, 0.05, 0.10],
                          learning_rate=['adaptive'],
                          activation=['logistic', 'relu'],
                          alpha=[0.1, 0.2],
                          early_stopping=[True, False],
                          n_iter_no_change=[20, 40],
                          random_state=[161227]
                          )
    regr_MLP = GridSearchCV(model_MLP,
                            param_grid_MLP,
                            scoring='r2')


regrs = [regr_models.regr_KNN,  # √
         regr_models.regr_MLP,  # √
         regr_models.regr_SVR,  # √
         regr_models.regr_XGB  # √
         ]

y_true = pd.DataFrame()
y_p_0 = pd.DataFrame()
y_p_1 = pd.DataFrame()
y_p_2 = pd.DataFrame()
y_p_3 = pd.DataFrame()
y_p_all = pd.DataFrame()

y_true_test = pd.DataFrame()
y_p_0_test = pd.DataFrame()
y_p_1_test = pd.DataFrame()
y_p_2_test = pd.DataFrame()
y_p_3_test = pd.DataFrame()
y_p_all_test = pd.DataFrame()

for i in range(0, int(max(all_features[method])) + 1, 1):
    dataset = all_features[all_features[method] == i]
    X = minmax_scale(dataset.ix[:, :].drop(['chla'], axis=1))
    y = dataset['chla'].as_matrix()
    print('Sub-Region', i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=161227)

    skf = list(KFold(n_splits=4, shuffle=True, random_state=161227).split(X=X_train, y=y_train))
    blend_train = np.zeros((X_train.shape[0], len(regrs)))
    blend_test = np.zeros((X_test.shape[0], len(regrs)))

    y_true = pd.concat([pd.Series(y_true), pd.Series(y_train)], axis=0)
    y_true_test = pd.concat([pd.Series(y_true_test), pd.Series(y_test)], axis=0)

    for j, regr in enumerate(regrs):
        print('Training Regressor ', j)
        blend_test_j = np.zeros((X_test.shape[0], len(skf)))
        for i, (cv_train_index, cv_test_index) in enumerate(skf):
            # print('Fold [%s]' % (i))
            cv_train_X = X_train[cv_train_index]
            cv_test_X = X_train[cv_test_index]
            cv_train_y = y_train[cv_train_index]
            cv_test_y = y_train[cv_test_index]
            regr.fit(cv_train_X, cv_train_y)
            blend_train[cv_train_index, j] = regr.predict(cv_train_X)
            blend_test_j[:, i] = regr.predict(X_test)
            print(regr.best_params_)
        blend_test[:, j] = blend_test_j.mean(axis=1)  # ！
        # print(blend_train[:, j])
        print('Train R2:%f' % round(r2_score(y_train, blend_train[:, j]), 3))
        print('Test R2:%f' % round(r2_score(y_test, blend_test[:, j]), 3))
        print('Train MAE:%f' % round(mean_absolute_error(y_train, blend_train[:, j]), 3))
        print('Test MAE:%f' % round(mean_absolute_error(y_test, blend_test[:, j]), 3))
        print('Train MSE:%f' % round(mean_squared_error(y_train, blend_train[:, j]), 3))
        print('Test MSE:%f' % round(mean_squared_error(y_test, blend_test[:, j]), 3))

        # print('y_p_' + str(j))
        locals()['y_p_' + str(j)] = pd.concat([locals().get('y_p_' + str(j)), pd.Series(blend_train[:, j])], axis=0)
        locals()['y_p_' + str(j) + '_test'] = pd.concat(
            [locals().get('y_p_' + str(j) + '_test'), pd.Series(blend_test[:, j])], axis=0)
        # print('y_p_' + str(j))
        # print(y_p_0)

    regr_2 = regr_models.regr_RF
    regr_2.fit(blend_train, y_train)
    # print(regr_2.best_params_)
    print('Stacking')
    print('Train R2:%f' % round(r2_score(y_train, regr_2.predict(blend_train)), 3))
    print('Test R2:%f' % round(r2_score(y_test, regr_2.predict(blend_test)), 3))
    print('Train MAE:%f' % round(mean_absolute_error(y_train, regr_2.predict(blend_train)), 3))
    print('Test MAE:%f' % round(mean_absolute_error(y_test, regr_2.predict(blend_test)), 3))
    print('Train MSE:%f' % round(mean_squared_error(y_train, regr_2.predict(blend_train)), 3))
    print('Test MSE:%f' % round(mean_squared_error(y_test, regr_2.predict(blend_test)), 3))

    y_p_all = pd.concat([y_p_all, pd.Series(regr_2.predict(blend_train))], axis=0)
    y_p_all_test = pd.concat([y_p_all_test, pd.Series(regr_2.predict(blend_test))], axis=0)

y_trueDF = pd.concat([y_true, y_p_0, y_p_1, y_p_2, y_p_3, y_p_all], axis=1)
y_true_testDF = pd.concat([y_true_test, y_p_0_test, y_p_1_test, y_p_2_test, y_p_3_test, y_p_all_test], axis=1)

y_trueDF.to_csv('y_trueDF.csv', index=False)
y_true_testDF.to_csv('y_true_testDF.csv', index=False)

print('M0')
print(r2_score(y_true, y_p_0))
print(r2_score(y_true_test, y_p_0_test))
print(mean_absolute_error(y_true, y_p_0))
print(mean_absolute_error(y_true_test, y_p_0_test))
print(mean_squared_error(y_true, y_p_0))
print(mean_squared_error(y_true_test, y_p_0_test))

print('M1')
print(r2_score(y_true, y_p_1))
print(r2_score(y_true_test, y_p_1_test))
print(mean_absolute_error(y_true, y_p_1))
print(mean_absolute_error(y_true_test, y_p_1_test))
print(mean_squared_error(y_true, y_p_1))
print(mean_squared_error(y_true_test, y_p_1_test))

print('M2')
print(r2_score(y_true, y_p_2))
print(r2_score(y_true_test, y_p_2_test))
print(mean_absolute_error(y_true, y_p_2))
print(mean_absolute_error(y_true_test, y_p_2_test))
print(mean_squared_error(y_true, y_p_2))
print(mean_squared_error(y_true_test, y_p_2_test))

print('M3')
print(r2_score(y_true, y_p_3))
print(r2_score(y_true_test, y_p_3_test))
print(mean_absolute_error(y_true, y_p_3))
print(mean_absolute_error(y_true_test, y_p_3_test))
print(mean_squared_error(y_true, y_p_3))
print(mean_squared_error(y_true_test, y_p_3_test))

print('Stacking')
print(r2_score(y_true, y_p_all))
print(r2_score(y_true_test, y_p_all_test))
print(mean_absolute_error(y_true, y_p_all))
print(mean_absolute_error(y_true_test, y_p_all_test))
print(mean_squared_error(y_true, y_p_all))
print(mean_squared_error(y_true_test, y_p_all_test))
