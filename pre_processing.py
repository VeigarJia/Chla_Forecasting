from __future__ import division

# -*- coding: utf-8 -*-
__author__ = 'Veigar'
'''Feature importance analyses.'''
import matplotlib

matplotlib.use('TkAgg')
from Chla_Forecasting.feature_selector import FeatureSelector
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
N0 = pd.read_csv('./2015-2018ALL/N0.csv', header=0)  # .fillna(0)
# L0 = pd.read_csv('./2015-2018ALL/L0.csv', header=0).fillna(0)
L1 = pd.read_csv('./2015-2018ALL/L1.csv', header=0)  # .fillna(0)
L2 = pd.read_csv('./2015-2018ALL/L2.csv', header=0)  # .fillna(0)
L3 = pd.read_csv('./2015-2018ALL/L3.csv', header=0)  # .fillna(0)
L4 = pd.read_csv('./2015-2018ALL/L4.csv', header=0)  # .fillna(0)

# join(
#     L0.set_index(['Station', 'Year', 'Season']),
#     on=['Station', 'Year', 'Season'],
#     how='left',
# ).
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
).dropna()  # Delete all the missing values

X = ALL.drop(['Chl-a'], axis=1)
y = ALL['Chl-a']

fs = FeatureSelector(data=X, labels=y)

# 1.Missing value greater than a certain percentage
fs.identify_missing(missing_threshold=0)

# 2.Collinearity greater than a certain threshold
fs.identify_collinear(correlation_threshold=0.99)
# fs.plot_collinear()

# 3.Zero importance feature
fs.identify_zero_importance()

# 4.View the number of features when the cumulative importance reaches the threshold (low importance features)
fs.plot_feature_importances(plot_n=20)
fs.identify_low_importance(cumulative_importance=0.99)
feature_importances_results = fs.feature_importances

# Write feature importance results to file
# print(feature_importances_results)
# pd.DataFrame(feature_importances_results).to_csv('Feature_Importance.csv')

# 5.Unique value feature
fs.identify_single_unique()

# 6.Delete filtered features
remain_X = fs.remove(methods='all',  # ['missing', 'collinear', 'zero_importance', 'low_importance', 'single_unique']
                     keep_one_hot=True)
print(len(remain_X))
# pd.concat([round(remain_X, 3), pd.DataFrame({'chla': y})], axis=1).to_csv('Feature_Selection.csv', index=False)
