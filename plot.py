import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

import h5py


"""
Code for creating manifold later!

z = np.matrix([sp.norm.ppf(gridValues[i]),sp.norm.ppf(gridValues[j])]).T
if continuous:
    h_decoder = np.log(1 + np.exp(np.dot(W4,z) + b4))
    y = 1 / (1 + np.exp(-(W5.dot(h_decoder) + b5)))
else:
    h_encoder = np.tanh(W4.dot(z) + b4)
    y = 1 / (1 + np.exp(-(W5.dot(h_encoder) + b5)))
"""

def plotdigits(numcols):
    f = h5py.File('datasets/mnist.hdf5','r')
    data = np.array(f["x_train"])

    shape = (28,28)

    columns = np.arange(0,numcols**2,numcols)

    image = np.vstack([np.hstack([data[i+j].reshape(shape) for j in xrange(numcols)]) for i in columns])
    
    plt.imshow(image, interpolation='nearest', cmap='Greys')
    plt.axis('off')
    plt.show()

plotdigits(5)

