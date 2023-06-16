import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

x1 = np.random.randn(1000)
x2 = np.random.randn(1000)

x = np.stack((x1, x2), axis=1)
print(x.shape)

kmeans = KMeans(n_clusters=4, random_state=7, max_iter=1000)
kmeans.fit(x)

labels = kmeans.labels_
print(labels)
print(len(labels))

plt.scatter(x[:, 0], x[:, 1], c=labels, s=60, alpha=0.8)
plt.show()

# ax.scatter(x[:, 0], x[:, 1], c=labels, s=60, alpha=0.8)
# plt.show()

tsne = TSNE(perplexity=40)
z = tsne.fit_transform(x)

plt.scatter(z[:, 0], z[:, 1], c=labels, s=60, alpha=0.8)
plt.show()