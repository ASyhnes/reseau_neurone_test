import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make.blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print(" dimension de X : ", X.shape)
print(" dimension de y : ", y.shape)
plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
plt.show()