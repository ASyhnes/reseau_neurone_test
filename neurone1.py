import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print("dimensions de x:", x.shape)
print("dimensions de y:", y.shape)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='summer')
plt.show()