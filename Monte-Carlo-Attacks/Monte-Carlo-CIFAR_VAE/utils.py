from keras.datasets import cifar10
from cifar10_params import *
from keras import backend as K
import numpy as np

def load_cifar10_with_validation(percentage=0.1, reuse=False):
    # tensorflow uses channels_last
    # theano uses channels_first
    if K.image_data_format() == 'channels_first':
        original_img_size = (img_chns, img_rows, img_cols)
    else:
        original_img_size = (img_rows, img_cols, img_chns)

    (x_train, _), (x_test, y_test) = cifar10.load_data()
    x_train = np.concatenate((x_train, x_test), axis=0)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)

    n = 60000
    border = int(n*percentage)

    train_inds = None
    if reuse:
        train_inds = np.loadtxt('train_inds.csv').astype(int)
    else:
        train_inds = np.arange(len(x_train))
        np.random.shuffle(train_inds)

    x_train = x_train[train_inds]

    np.savetxt('train_inds.csv', train_inds)
    np.savetxt('percentage.csv', [percentage])
    print('###########')
    print(len(x_train[:border]))
    print(len(x_train[border:]))
    print('###########')
    
    return x_train[:border], x_train[border:]

def load_cifar10_with_validation_flattened(percentage=0.1, reuse=False):

    x_train, x_test = load_cifar10_with_validation(percentage, reuse)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    return x_train.reshape(len(x_train),-1), x_test.reshape(len(x_test),-1)