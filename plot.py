import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

import h5py


"""
Code for creating manifold later!
if continuous:
        h_decoder = np.log(1 + np.exp(np.dot(W4,z) + b4))
        y = 1 / (1 + np.exp(-(W5.dot(h_decoder) + b5)))
"""

def manifold(gridSize, epoch):
    f = h5py.File('params/epoch_' + str(epoch) + '.hdf5','r')

    shape = (28,28)

    wtanh = np.matrix(f["wtanh"])
    btanh = np.matrix(f["btanh"]).T
    wsig = np.matrix(f["wsig"])
    bsig = np.matrix(f["bsig"]).T

    wb = (wtanh,btanh,wsig,bsig)
    gridValues = np.linspace(0.05,0.95,gridSize)

    #Create matrix of z instead of with lambda
    z = lambda i,j: np.matrix([sp.norm.ppf(gridValues[i]),sp.norm.ppf(gridValues[j])]).T

    image = np.vstack([np.hstack([activation_binary(z(i,j),wb).reshape(shape) for j in xrange(gridSize)]) for i in xrange(gridSize)])

    plt.imshow(image, interpolation='nearest', cmap='Greys')
    plt.axis('off')
    plt.show()

def activation_binary(z, wb):
    wtanh, btanh, wsig, bsig = wb

    h = np.tanh(wtanh.dot(z) + btanh)
    y = 1 / (1 + np.exp(-(wsig.dot(h) + bsig)))

    return y

def plotdigits(numcols):
    f = h5py.File('datasets/mnist.hdf5','r')
    data = np.array(f["x_train"])

    shape = (28,28)

    columns = np.arange(0,numcols**2,numcols)

    image = np.vstack([np.hstack([data[i+j].reshape(shape) for j in xrange(numcols)]) for i in columns])
    
    plt.imshow(image, interpolation='nearest', cmap='Greys')
    plt.axis('off')
    plt.show()

manifold(10,2)
