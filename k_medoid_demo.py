'''
Implementation of K-medoids

'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

plt.style.use('bmh')
plt.rcParams.update({'font.size': 18})

# helper for visualization
def visualize_step(ax,j,X,labels):
    ii = j//ax.shape[1]
    jj = j%ax.shape[0]
    
    ax[ii,jj].scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Accent)
    ax[ii,jj].scatter(X[medoids,0], X[medoids,1], c=labels[medoids], s=200, edgecolor='k', lw=2, cmap=plt.cm.Accent)
    
    ax[ii,jj].text(0.05, 0.95, 'iter=%i'%j,  ha='left', va='top', fontsize=20, bbox={'color':[0.8,0.8,0.8]}, transform=ax[ii,jj].transAxes)
    
    return

# reproducibility of RNG
np.random.seed(7716)

##########
# Generate synthetic data.
p = 100

centers = np.array([
    [0,0],
    [3,3],
    [-1,4]
    ])

X = np.concatenate([
    centers[0] + np.random.randn(p,2),
    centers[1] + np.random.randn(p,2),
    centers[2] + np.random.randn(p,2),
    ])

# shuffle data
X = X[np.random.permutation(X.shape[0])]

fig,ax = plt.subplots(3,3, sharex=True, sharey=True, figsize=(10,10), constrained_layout=True)

#################################
# Pretend we don't know clusters.

D = metrics.pairwise_distances(X, metric='euclidean')

# random guess.
K = 3
medoids = np.random.permutation(X.shape[0])[:K]
labels = np.argmin( D[medoids, :] , axis=0 )

for j in range(np.prod(ax.shape)):

    groups = [np.where(labels==i)[0] for i in range(len(medoids))]
    for i in range(len(medoids)):
        medoids[i] = np.argmin( np.sum( D[groups[i]] , axis=0) )
    
    labels = np.argmin( D[medoids, :] , axis=0 )
    visualize_step(ax, j, X, labels)

fig.savefig('medoid_vis.png')
