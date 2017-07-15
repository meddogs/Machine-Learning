#Script to show how PCA works
import numpy as np
import matplotlib.pyplot as mplot
from sklearn.decomposition import PCA
pca = PCA()
# generate an N = 500, D = 5 data matrix
X = np.random.random((500,5))
Z = pca.fit_transform(X)
# Display the covariance of Z
mplot.imshow(np.cov(Z.T))
mplot.show()
