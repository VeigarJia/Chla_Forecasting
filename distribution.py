# -*- coding: utf-8 -*-
__author__ = 'Veigar'
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from pylab import mpl
from sklearn.preprocessing import *

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


# Create models from data
def best_fit_distribution(data, bins, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [st.norm]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体  
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

origin = pd.read_csv('特征选择.csv')
others = pd.read_csv('聚类结果.csv')
all_features = origin.join(
    others.set_index(['x', 'y']),
    on=['x', 'y'],
    how='left',
)

method = 'KMeans'  # 'AgglomerativeClustering','KMeans','MiniBatchKMeans','GM'
y_test_origin = []
y_test_predict = []
y_train_origin = []
y_train_predict = []
for i in range(2, 3, 1):
    dataset = all_features[all_features[method] == i]
    n = 13
    X = dataset.ix[:, n]

    X_scale = pd.Series(minmax_scale(dataset.ix[:, n]))

    data = X
    # data = X_scale

    bins = np.arange(min(data), max(data), (max(data) - min(data)) / 100)
    plt.figure(figsize=(7, 4))
    ax = data.plot(kind='hist', bins=bins, density=True, alpha=0.5)
    dataYLim = ax.get_ylim()
    best_fit_name, best_fit_params = best_fit_distribution(data=data, bins=bins, ax=ax)
    best_dist = getattr(st, best_fit_name)
    ax.set_ylim(dataYLim)

    # ax.set_xlabel(u'COD (0-1 Standardization)')
    ax.set_xlabel(u'COD')

    ax.set_ylabel('Density')
    pdf = make_pdf(best_dist, best_fit_params)
    plt.savefig('./pictures/Distribution 0.jpg', dpi=400)
    plt.show()
