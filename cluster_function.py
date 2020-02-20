# -*- coding: utf-8 -*-
__author__ = 'Veigar'
'Write cluster results to file for calling'
'Draw clustering results and display them as coordinate points'

from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabaz_score, silhouette_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
dataset = pd.read_csv('2015-2018Time.csv', header=0, encoding='GBK').dropna()


def plot_data(set, predicted_labels):
    labels = np.unique(predicted_labels)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    colors = ['#003366', '#009933', '#0099ff', '#993333', '#cc66ff', '#ff33cc', '#00ffcc', '#ffff66', '#ff0000']

    for i, label in enumerate(labels):
        position = predicted_labels == label
        ax.scatter(set.ix[position, 0], set.ix[position, 1], label=label,
                   c=colors[i % len(colors)], )
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    plt.legend()
    plt.show()


X = dataset.ix[:, 3:]

settle = dataset.ix[:, 1:3]
station_name = dataset.ix[:, 0]
station_x = dataset.ix[:, 1]
station_y = dataset.ix[:, 2]

clus_KMeans = cluster.KMeans(n_clusters=7, random_state=161227)
clus_KMeans.fit(X)
predicted_label_KMeans = clus_KMeans.fit_predict(X)

clus_AgglomerativeClustering = cluster.AgglomerativeClustering(n_clusters=6)
clus_AgglomerativeClustering.fit(X)
predicted_label_AgglomerativeClustering = clus_AgglomerativeClustering.fit_predict(X)

clus_MiniBatchKMeans = cluster.MiniBatchKMeans(n_clusters=5, random_state=161227)
clus_MiniBatchKMeans.fit(X)
predicted_label_MiniBatchKMeans = clus_MiniBatchKMeans.fit_predict(X)

clus_GM = GaussianMixture(n_components=5, random_state=161227)
clus_GM.fit(X)
predicted_label_GM = clus_GM.fit_predict(X)

cluster_class = pd.DataFrame({
    'Station': station_name,
    'x': station_x,
    'y': station_y,
    'KMeans': predicted_label_KMeans,
    'AgglomerativeClustering': predicted_label_AgglomerativeClustering,
    'MiniBatchKMeans': predicted_label_MiniBatchKMeans,
    'GM': predicted_label_GM
})

cluster_class.to_csv('Cluster_Results.csv', index=False)

# plot_data(settle, predicted_label_KMeans)
# plot_data(settle, predicted_label_AgglomerativeClustering)
# plot_data(settle, predicted_label_MiniBatchKMeans)
# plot_data(settle, predicted_label_GM)

# Selecting parameters of various clustering algorithms
score_KMeans = []
score_AgglomerativeClustering = []
score_MiniBatchKMeans = []
score_GM = []
n_clusters = np.arange(2, 14)

for n_cluster in n_clusters:
    clus_KMeans = cluster.KMeans(n_clusters=n_cluster, random_state=161227)
    clus_KMeans.fit(X)
    predicted_label_KMeans = clus_KMeans.fit_predict(X)
    score_KMeans.append(calinski_harabaz_score(X, predicted_label_KMeans))

    clus_AgglomerativeClustering = cluster.AgglomerativeClustering(n_clusters=n_cluster)
    clus_AgglomerativeClustering.fit(X)
    predicted_label_AgglomerativeClustering = clus_AgglomerativeClustering.fit_predict(X)
    score_AgglomerativeClustering.append(calinski_harabaz_score(X, predicted_label_AgglomerativeClustering))

    clus_GM = GaussianMixture(n_components=n_cluster, random_state=161227)
    clus_GM.fit(X)
    predicted_label_GM = clus_GM.fit_predict(X)
    score_GM.append(calinski_harabaz_score(X, predicted_label_GM))

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
l1, = plt.plot(n_clusters, score_KMeans, label='K-means Cluster')
o1, = plt.plot(np.argmax(score_KMeans) + 2, score_KMeans[np.argmax(score_KMeans)], 'ro', label='The Best Cluster Model')
l2, = plt.plot(n_clusters, score_AgglomerativeClustering, label='Agglomerative Cluster')
l3, = plt.plot(n_clusters, score_GM, label='Gaussian Mixture Cluster')
plt.xlim((1, 14))
plt.ylim((100, 180))

legend1 = plt.legend(handles=[l1, l2, l3, o1], loc=0, scatterpoints=1)
plt.gca().add_artist(legend1)
plt.tight_layout()
plt.savefig('./pictures/scatter parameter.jpg', dpi=400)
plt.show()
