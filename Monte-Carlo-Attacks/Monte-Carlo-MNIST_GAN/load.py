import sys
sys.path.append('..')

import numpy as np
import os

def mnist(data_dir):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY

def mnist_with_valid_set(percentage, data_dir):
    trX, teX, trY, teY = mnist(data_dir)

    train_inds = np.arange(len(trX))
    np.random.shuffle(train_inds)
    trX = trX[train_inds]
    trY = trY[train_inds]
    #trX, trY = shuffle(trX, trY)
    n = 60000
    border = int(n*percentage)
    vaX = trX[border:]
    vaY = trY[border:]
    trX = trX[:border]
    trY = trY[:border]

    np.savetxt('train_inds.csv', train_inds)
    np.savetxt('percentage.csv', [percentage])

    return trX, vaX, teX, trY, vaY, teY


def load_mnist_with_valid_set(train_inds, percentage, data_dir):
    trX, teX, trY, teY = mnist(data_dir)

    trX = trX[train_inds]
    trY = trY[train_inds]
    #trX, trY = shuffle(trX, trY)
    n = 60000
    border = int(n*percentage)
    vaX = trX[border:]
    vaY = trY[border:]
    trX = trX[:border]
    trY = trY[:border]

    return trX, vaX, teX, trY, vaY, teY
