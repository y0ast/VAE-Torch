import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

import h5py

def manifold(gridSize, binary, epoch):
    f = h5py.File('params/ff_epoch_' + str(epoch) + '.hdf5','r')

    wsig = np.matrix(f["wsig"])
    bsig = np.matrix(f["bsig"]).T

    if binary:
        shape = (28,28)
        activation = lambda z, wb : activation_binary(z,wb)
        wtanh = np.matrix(f["wtanh"])
        btanh = np.matrix(f["btanh"]).T
        wb = (wtanh,btanh,wsig,bsig)
    else:
        shape = (28,20)
        activation = lambda z, wb: activation_continuous(z,wb)
        wrelu = np.matrix(f["wrelu"])
        brelu = np.matrix(f["brelu"]).T
        wb = (wrelu,brelu,wsig,bsig)

    gridValues = np.linspace(0.05,0.95,gridSize)

    z = lambda gridpoint: np.matrix(sp.norm.ppf(gridpoint)).T


    image = np.vstack([np.hstack([activation(z((i,j)),wb).reshape(shape) for j in gridValues]) for i in gridValues])

    plt.imshow(image, cmap='Greys')
    plt.axis('off')
    plt.show()


def activation_binary(z, wb):
    wtanh, btanh, wsig, bsig = wb

    h = np.tanh(wtanh.dot(z) + btanh)
    y = 1 / (1 + np.exp(-(wsig.dot(h) + bsig)))

    return y

def activation_continuous(z, wb):
    wrelu, brelu, wsig, bsig = wb

    h = np.log(1 + np.exp(np.dot(wrelu,z) + brelu))
    y = 1 / (1 + np.exp(-(np.dot(wsig,h) + bsig)))

    return y


#Gridsize, Binary (True/False), Epoch number for params
manifold(10,False,740)
