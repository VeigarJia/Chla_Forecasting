# Chla-Forecasting
DESCRIPTION
2015-2018ALL: dataset used in the experiment. LK represents the K-season-behind historical data of water quality monitoring stations. Due to the confidentiality of the data, we hash the number of monitoring stations and hide the geographical location coordinates of the stations.
Pictures: experimental results. 
2015-2018Time.csv: the seasonal average concentration of chlorophyll-a of each monitoring station in 2015-2018.
Cluster_function.py: group stations through various clustering algorithms.
Different_model.py: regression learners.
Distribution.py: obtain a vector distribution and normal distribution fitting function.
Feature_selector.py: feature selection.
Importance_analyse.py: a basic learner for calculating the importance of features.
Plot_feature_importances.py: draw feature importance figures.
Plot_result_scatter.py: draw scatter map according to clustering results.
Pre_processing.py: data preprocessing.
Stacking_model_no_cluster.py: Stacking-based approach to forecasting Chl-a concentration.
Stacking_model_print_result.py: Cluster-Stacking-based approach to forecasting Chl-a concentration.

HOW TO USE
1.	Cluster monitoring stations by cluster_function.py. 
2.	Plot the clustering results by plot_result_scatter.py (optional).
3.	Data preprocessing through pre processing.py
4.	Draw the result of feature importance analysis by plot feature imports.py.
5.	Achieve stacking-based approach through Stacking_model_no_cluster.py.
6.	Achieve Cluster-Stacking-based approach through Stacking_model_print_result.py.
