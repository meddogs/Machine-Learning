# Kmeans using dataset from sklearn
# Load libraries and dataset from sklearn
from sklearn.cluster import  KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as mplot

data = load_iris()
features = data.data
labels = data.target 

# Here we use initialize and fit the features

kft = KMeans(n_clusters=3, init='k-means++', max_iter=999, n_init=1, random_state=101)
kft.fit(features)

#plot the grid
mplot.scatter(features[:,0], features[:,1], s=100, c=labels,
            edgecolors='white', alpha=0.85, cmap='autumn')
mplot.grid()
mplot.xlabel(data.feature_names[0]) #  x axis label
mplot.ylabel(data.feature_names[1]) # y axis label
# Display the centroids for KMeans
mplot.scatter(kft.cluster_centers_[:,0], kft.cluster_centers_[:,1],
            s=150, marker = "x", c = ["blue","green","red"])

for class_no in range(0,3):
    mplot.annotate(data.target_names[class_no],
          (features[3+50*class_no,0],features[3+50*class_no,1]))
# Display the output
mplot.show()