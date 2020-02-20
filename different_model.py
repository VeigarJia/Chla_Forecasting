# -*- coding: utf-8 -*-
__author__ = 'Veigar'
from math import *
import pandas as pd
import numpy as np
import xgboost
import matplotlib

matplotlib.use('TkAgg')
from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
import warnings

warnings.filterwarnings("ignore")


def test_XGBoost_KFlod(i, X_train, X_test, y_train, y_test):
    kfold = KFold(n_splits=3, random_state=161227)
    model = xgboost.XGBRegressor()
    param_grid = dict(
        learning_rate=[0.2],
        max_depth=[4],
        seed=[1227])
    fit_params = dict(early_stopping_rounds=20, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    regr = GridSearchCV(model, param_grid, verbose=False, scoring='neg_mean_absolute_error',
                        fit_params=fit_params,
                        cv=kfold, n_jobs=-1)
    XGBoost = regr.fit(X_train, y_train)
    joblib.dump(XGBoost, './model/xgb_' + str(i) + '.model')

    print('best para: ', regr.best_params_)

    y_pred_train = regr.predict(X_train)
    print('train')
    print(f'R2: {round(r2_score(y_train,y_pred_train),3)}')
    print(f'MAE:{round(mean_absolute_error(y_train,y_pred_train),3)}')
    print(f'MSE:{round(mean_squared_error(y_train,y_pred_train),3)}')

    y_pred_test = regr.predict(X_test)
    print('test')
    print(f'R2: {round(r2_score(y_test,y_pred_test),3)}')
    print(f'MAE:{round(mean_absolute_error(y_test,y_pred_test),3)}')
    print(f'MSE:{round(mean_squared_error(y_test,y_pred_test),3)}')


def test_RF_KFlod(i, X_train, X_test, y_train, y_test):
    kfold = KFold(n_splits=3, random_state=161227)
    model = RandomForestRegressor()
    param_grid = dict(
        n_estimators=[30],
        max_depth=[5, 10],
        min_samples_split=[4],
        random_state=[161227]
    )
    regr = GridSearchCV(model, param_grid, cv=kfold, scoring='r2')
    RF = regr.fit(X_train, y_train)
    joblib.dump(RF, './model/rf_' + str(i) + '.model')
    print('best para: ', regr.best_params_)

    y_pred_train = regr.predict(X_train)
    print('train')
    print(f'R2: {round(r2_score(y_train,y_pred_train),3)}')
    print(f'MAE:{round(mean_absolute_error(y_train,y_pred_train),3)}')
    print(f'MSE:{round(mean_squared_error(y_train,y_pred_train),3)}')

    y_pred_test = regr.predict(X_test)
    print('test')
    print(f'R2: {round(r2_score(y_test,y_pred_test),3)}')
    print(f'MAE:{round(mean_absolute_error(y_test,y_pred_test),3)}')
    print(f'MSE:{round(mean_squared_error(y_test,y_pred_test),3)}')


def test_AdaBoost_KFlod(i, X_train, X_test, y_train, y_test):
    kfold = KFold(n_splits=4, random_state=161227)
    model = AdaBoostRegressor()
    param_grid = dict(
        n_estimators=[30, 50],
        learning_rate=[0.2, 0.5, 0.8, 1],
        random_state=[161227]
    )
    regr = GridSearchCV(model, param_grid, cv=kfold, scoring='r2')
    RF = regr.fit(X_train, y_train)
    joblib.dump(RF, './model/rf_' + str(i) + '.model')
    print('best para: ', regr.best_params_)

    y_pred_train = regr.predict(X_train)
    print('train')
    print(f'R2: {round(r2_score(y_train,y_pred_train),3)}')
    print(f'MAE:{round(mean_absolute_error(y_train,y_pred_train),3)}')
    print(f'MSE:{round(mean_squared_error(y_train,y_pred_train),3)}')

    y_pred_test = regr.predict(X_test)
    print('test')
    print(f'R2: {round(r2_score(y_test,y_pred_test),3)}')
    print(f'MAE:{round(mean_absolute_error(y_test,y_pred_test),3)}')
    print(f'MSE:{round(mean_squared_error(y_test,y_pred_test),3)}')


def test_Tree_KFlod(i, X_train, X_test, y_train, y_test):
    kfold = KFold(n_splits=4, random_state=161227)
    model = DecisionTreeRegressor()
    param_grid = dict(min_samples_leaf=[3],
                      random_state=[161227])
    regr = GridSearchCV(model, param_grid, cv=kfold, scoring='r2')
    Tree = regr.fit(X_train, y_train)

    print('best para: ', regr.best_params_)
    joblib.dump(Tree, './model/tree_' + str(i) + '.model')
    y_pred_train = regr.predict(X_train)
    print('train')
    print(f'R2: {round(r2_score(y_train,y_pred_train),3)}')
    print(f'MAE:{round(mean_absolute_error(y_train,y_pred_train),3)}')
    print(f'MSE:{round(mean_squared_error(y_train,y_pred_train),3)}')

    y_pred_test = regr.predict(X_test)
    print('test')
    print(f'R2: {round(r2_score(y_test,y_pred_test),3)}')
    print(f'MAE:{round(mean_absolute_error(y_test,y_pred_test),3)}')
    print(f'MSE:{round(mean_squared_error(y_test,y_pred_test),3)}')


def test_KNN_KFlod(i, X_train, X_test, y_train, y_test):
    kfold = KFold(n_splits=3, random_state=161227)
    model = KNeighborsRegressor()
    param_grid = dict(weights=['distance'],
                      p=[1],
                      n_neighbors=[2],
                      )
    regr = GridSearchCV(model, param_grid, cv=kfold, scoring='r2')
    KNN = regr.fit(X_train, y_train)
    joblib.dump(KNN, './model/knn_' + str(i) + '.model')
    print('best para: ', regr.best_params_)

    y_pred_train = regr.predict(X_train)
    print('train')
    print(f'R2: {round(r2_score(y_train,y_pred_train),3)}')
    print(f'MAE:{round(mean_absolute_error(y_train,y_pred_train),3)}')
    print(f'MSE:{round(mean_squared_error(y_train,y_pred_train),3)}')

    y_pred_test = regr.predict(X_test)
    print('test')
    print(f'R2: {round(r2_score(y_test,y_pred_test),3)}')
    print(f'MAE:{round(mean_absolute_error(y_test,y_pred_test),3)}')
    print(f'MSE:{round(mean_squared_error(y_test,y_pred_test),3)}')


def test_SVR_KFlod(i, X_train, X_test, y_train, y_test):
    kfold = KFold(n_splits=4, random_state=161227)
    model = svm.SVR()
    param_grid = dict(C=np.logspace(-1, 2, 10),
                      kernel=['rbf'],
                      epsilon=[0.02])
    svr = GridSearchCV(model, param_grid, cv=kfold, scoring='r2')
    svr.fit(X_train, y_train)
    joblib.dump(svr, './model/svr_' + str(i) + '.model')
    print('best para: ', svr.best_params_)

    y_pred_train = svr.predict(X_train)
    print('train')
    print(f'R2: {round(r2_score(y_train,y_pred_train),3)}')
    print(f'MAE:{round(mean_absolute_error(y_train,y_pred_train),3)}')
    print(f'MSE:{round(mean_squared_error(y_train,y_pred_train),3)}')

    y_pred_test = svr.predict(X_test)
    print('test')
    print(f'R2: {round(r2_score(y_test,y_pred_test),3)}')
    print(f'MAE:{round(mean_absolute_error(y_test,y_pred_test),3)}')
    print(f'MSE:{round(mean_squared_error(y_test,y_pred_test),3)}')


def test_LASSO_KFlod(X_train, X_test, y_train, y_test):
    alpha = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    kfold = KFold(n_splits=4, random_state=161227)
    model = Lasso()
    param_grid = dict(alpha=alpha)
    regr = GridSearchCV(model, param_grid, cv=kfold, scoring='r2')
    regr.fit(X_train, y_train)

    print('best para: ', regr.best_params_)

    y_pred_train = regr.predict(X_train)
    print('train')
    print(f'R2: {r2_score(y_train,y_pred_train)}')
    print(f'MAE:{mean_absolute_error(y_train,y_pred_train)}')
    print(f'MSE:{mean_squared_error(y_train,y_pred_train)}')

    y_pred_test = regr.predict(X_test)
    print('test')
    print(f'R2: {r2_score(y_test,y_pred_test)}')
    print(f'MAE:{mean_absolute_error(y_test,y_pred_test)}')
    print(f'MSE:{mean_squared_error(y_test,y_pred_test)}')
