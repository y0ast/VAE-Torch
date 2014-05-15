#Simple file that converts pickles to hdf5 files.

import numpy as np

import cPickle, h5py

datafile = open('freyfaces.pkl', 'rb')
data = cPickle.load(datafile)

np.random.shuffle(data)

train = data[:1500]
test = data[1500:]

print train.shape

hdffile = h5py.File('freyfaces.hdf5','w')
hdffile.create_dataset("train",data=train)
hdffile.create_dataset("test",data=test)

