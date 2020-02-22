# PROJECT INTRODUCTION
This is A Cluster-Stacking-based Approach to Forecasting Seasonal Chlorophyll-a Concentration in Coastal Waters

# FILE DESCRIPTION
1.	2015-2018ALL: The folder used to store the original data of the experiment. Recommended file naming: LK.csv, where K represents the K-season-behind historical data of water quality monitoring stations. Due to the confidentiality of marine monitoring station information, we do not provide the original data set for the experiment.
2.	Pictures: experimental results. 
3.	2015-2018Time.csv: the seasonal average concentration of chlorophyll-a of each monitoring station in 2015-2018.
4.	Cluster_function.py: group stations through various clustering algorithms.
5.	Different_model.py: regression learners.
6.	Distribution.py: obtain a vector distribution and normal distribution fitting function.
7.	Feature_selector.py: feature selection.
8.	Importance_analyse.py: a basic learner for calculating the importance of features.
9.	Plot_feature_importances.py: draw feature importance figures.
10.	Plot_result_scatter.py: draw scatter map according to clustering results.
11.	Pre_processing.py: data preprocessing.
12.	Stacking_model_no_cluster.py: Stacking-based approach to forecasting Chl-a concentration.
13.	Stacking_model_print_result.py: Cluster-Stacking-based approach to forecasting Chl-a concentration.

# HOW TO USE
1.	Cluster monitoring stations by cluster_function.py. 
2.	Plot the clustering results by plot_result_scatter.py (optional).
3.	Data preprocessing through pre processing.py
4.	Draw the result of feature importance analysis by plot_feature_importants.py (optional).
5.	Achieve stacking-based approach through Stacking_model_no_cluster.py.
6.	Achieve Cluster-Stacking-based approach through Stacking_model_print_result.py.

# NOTE
Some of the intermediate documents are not shown in this project.
