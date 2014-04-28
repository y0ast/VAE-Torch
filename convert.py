#Simple file that converts pickles to hdf5 files.

import numpy as np

import cPickle, h5py

datafile = open('datafile', 'rb')
data = cPickle.load(datafile)

hdffile = h5py.File('datafile.hdf5','w')
hdffile.create_dataset("data",data=data)
hdffile.create_dataset("labels",data=labels)

