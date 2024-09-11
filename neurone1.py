import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Génération des données
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print("dimensions de X:", X.shape)
print("dimensions de y:", y.shape)

# Affichage des données
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.show()

# Fonction d'initialisation
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

# Initialisation avec la variable 'X' 
W, b = initialisation(X)
print(b.shape)

# Def Model
def model(X, W, b):
    Z = X.dot(W) + b
