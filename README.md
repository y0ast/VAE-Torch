##Variational Auto-encoder

This is an implementation of the paper [Stochastic Gradient VB and the Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by D. Kingma and Prof. Dr. M. Welling.

In my other [repository](https://github.com/y0ast/Variational-Autoencoder) the implementation is in Python (Theano), this version is based on Torch7.

To run the MNIST experiment:

`th binaryva.lua`

Note that the resolution of the MNIST digits is 32x32, while in the paper Kingma uses 28x28.

To run the FreyFace experiment:

`th continuousva.lua`

The code is MIT licensed.

