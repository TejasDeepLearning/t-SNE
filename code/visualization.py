import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE 

n_components = 5
X, y = make_blobs(n_samples=300, n_features=4, centers=n_components, random_state=7)

print(X.shape)
print(y.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], X[:, 3], c=y)
plt.show()

tsne = TSNE()
transformed = tsne.fit_transform(X)

plt.scatter(transformed[:, 0], transformed[:, 1], c=y)
plt.show()