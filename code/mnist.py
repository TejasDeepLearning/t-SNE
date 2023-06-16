import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from util import getKaggleMNIST

xtrain, ytrain, xtest, ytest = getKaggleMNIST()
print(xtrain.shape)
print(ytrain.shape)

# reduce the sample size because t-sne is going to crash if we dont
sample_size = 1000
X = xtrain[:sample_size]
Y = ytrain[:sample_size]

tsne = TSNE()
Z = tsne.fit_transform(X)
plt.scatter(Z[:, 0], Z[:, 1], s=90, c=Y, alpha=0.5)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xtrain[0], xtrain[0], c=ytrain)
plt.show()