# Script to demonstrate K means
import matplotlib.pyplot as mplot
from matplotlib import style
import numpy as npy
from sklearn.cluster import KMeans
style.use('ggplot')
# data
sampledata = npy.array([[0.5, 1.5],[1.2, 1.1],[3, 4],[5, 6], [7, 7],[6, 7.5],[2.3,3],[4.2, 5],[3.4,2.8],[5,6]])
kft = KMeans(n_clusters=3)
kft.fit(sampledata)
centroids = kft.cluster_centers_
labels = kft.labels_
colors = ["b.","g.","r."]
for i in range(len(sampledata)):
    mplot.plot(sampledata[i][0], sampledata[i][1], colors[labels[i]], markersize = 10)
mplot.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=100, linewidths = 5, zorder = 10)
mplot.show()

