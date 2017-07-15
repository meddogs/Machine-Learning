# Script to perform kmeans on Falmigos data set given k =4
# Load the required Libraries
import scipy as sp
import matplotlib.pyplot as mplot
from sklearn.cluster import KMeans
# use SciPy's genfromtxt(), to load the data from a csv file.
f_data = sp.genfromtxt("C:\Work\ML\csvsample.csv", delimiter=",")

# Initialise the number of clusters to 4
kft = KMeans(n_clusters=4)
kft.fit(f_data)
centroids = kft.cluster_centers_
labels = kft.labels_
# Configure other settings for the chart
colors = ["b.","g.","r.","y."]
mplot.title("Customer Transactions")
mplot.xlabel("Amount Spent")
mplot.ylabel("Vouchers Used")
for i in range(len(f_data)):
    mplot.plot(f_data[i][0], f_data[i][1], colors[labels[i]], markersize = 10)
mplot.scatter(centroids[:, 0],centroids[:, 1], marker = "x", c = "black", s=100, linewidths = 5, zorder = 10)
mplot.grid()
mplot.show()


