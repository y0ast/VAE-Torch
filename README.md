##Variational Auto-encoder

This is an improved implementation of the paper [Stochastic Gradient VB and the Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by D. Kingma and Prof. Dr. M. Welling. This code uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

In my other [repository](https://github.com/y0ast/Variational-Autoencoder) the implementation is in Python (Theano), this version is based on Torch7 and NNgraph.

To run the MNIST experiment:

`th main.lua`

Setting the continuous boolean to true will make the script run the freyfaces experiment.

The code is MIT licensed.

