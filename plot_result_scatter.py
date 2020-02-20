# -*- coding: utf-8 -*-
__author__ = 'Veigar'

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_absolute_error

# result_train = pd.read_csv('y_trueDF.csv')
result_train = pd.read_csv('y_true_testDF.csv')
# result_train = pd.read_csv('y_trueDF_whole.csv')
# result_train = pd.read_csv('y_true_testDF_whole.csv')
# result_train = pd.read_csv('y_trueDF_whole_nstation.csv')
# result_train = pd.read_csv('y_true_testDF_whole_nstation.csv')

x = result_train.ix[:, 0]  # True value
y = result_train.ix[:, 5]  # Results of different methods

print()
plt.figure(figsize=(4.5, 4))
# plt.rcParams['figure.dpi'] = 300
plt.scatter(x, y, s=3)
plt.plot(x, x, color='k', linewidth=0.5)
f = np.polyfit(x, y, deg=1)
plt.plot(x, f[0] * x + f[1], color='r', linewidth=0.5)
a = str('$\mathregular{R ^ 2}$=' + str(round(r2_score(x, y), 3)))
b = str('MAE=' + str(round(mean_absolute_error(x, y), 3)))
plt.text(x=2, y=37, s='(k)')
plt.text(x=2, y=34, s='Stacking (train)')
plt.text(x=2, y=31, s=a)
plt.text(x=2, y=28, s=b)
plt.xlim(-2, 42)
plt.ylim(-2, 42)
plt.xlabel('Observed Value')
plt.ylabel('Predicted Value')
plt.savefig('./pictures/(k) Stacking (train).jpg', dpi=400)
plt.show()
