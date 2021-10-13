# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:23:13 2021

@author: giris
"""

from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
iris = datasets.load_iris() 
#iris=pd.read_csv("C://data analytics lab//Iris.csv")
print(iris)
X=iris['data']
Y=iris.target
print(X.shape)

# pca 
X = PCA(n_components=2).fit_transform(X)
plt.scatter(X[:,0],X[:,1])

#agglomerative clustering module
from sklearn.cluster import AgglomerativeClustering 
classifier= AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'average') 
clusters = classifier.fit_predict(X)

plt.scatter(X[clusters == 0, 0], X[clusters == 0, 1], label = 'Type 1')
plt.scatter(X[clusters == 1, 0], X[clusters == 1, 1], label = 'Type 2')
plt.scatter(X[clusters == 2, 0], X[clusters == 2, 1], label = 'Type 3')
plt.title('Clusters')
plt.show()
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], label = 'Type 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], label = 'Type 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], label = 'Type 3')


linkage_type = "ward"
linkage_matrix=linkage(X[clusters],linkage_type)
plt.figure(figsize=(10,5))
dendrogram(linkage_matrix)
plt.show()