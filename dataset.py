import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimension de x:', x.shape)
print('dimension de y:', y.shape)

plt.scatter(x[:,0], x[:,1], c=y, cmap='summer')
plt.show()

def initialisation(x):
    w = np.random.randn(x.shape[1],1)
    b = np.random.randn(1)
    return(w,b)

w,b = initialisation(x)
b.shape