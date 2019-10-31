import os
import time 

import ais
import matplotlib.pyplot as plt
from priors import NormalPrior
from kernels import ParsenDensityEstimator
from scipy.stats import norm

import tensorflow as tf
import numpy as np
import mnist_data
import vae
import plot_utils
import glob
import sys
import time
import scipy
from sklearn.decomposition import PCA
from skimage.feature import hog

""" parameters """

#python gen_samples.py;

percentage = 0.1

dt = np.dtype([('instance_no', int),
               ('exp_no', int),
               ('method', int), # 1 = white box, 2 = euclidean_PCA, 3 = hog, 4 = euclidean_PCA category, 5 = hog category, 6 = ais
               ('pca_n', int),
               ('percentage_of_data', float),
               ('percentile', float),
               ('mc_euclidean_no_batches', int), # stuff
               ('mc_hog_no_batches', int), # stuff
               ('sigma_ais', float),
               ('11_perc_mc_attack_log', float),
               ('11_perc_mc_attack_eps', float),
               ('11_perc_mc_attack_frac', float), 
               ('50_perc_mc_attack_log', float), 
               ('50_perc_mc_attack_eps', float),
               ('50_perc_mc_attack_frac', float),
               ('50_perc_white_box', float),
               ('11_perc_white_box', float),
               ('50_perc_ais', float),
               ('50_perc_ais_acc_rate', float),
               ('successfull_set_attack_1', float),
               ('successfull_set_attack_2', float),
               ('successfull_set_attack_3', float)
              ])

IMAGE_SIZE_MNIST = 28
n_hidden = 500
dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image
dim_z = 10
model_no = '299'

""" prepare MNIST data """

train_total_data, train_size, valid_total_data, validation_size, test_total_data, test_size, _, _ = mnist_data.prepare_MNIST_data(reuse=True)
# compatibility with old attack
vaY = np.where(valid_total_data[:,784:795] == 1)[1]
trY = np.where(train_total_data[:,784:795] == 1)[1]
teY = np.where(test_total_data[:,784:795] == 1)[1]
vaX = valid_total_data[:,0:784]
trX = train_total_data[:,0:784]
teX = test_total_data[:,0:784]
n_samples = train_size

""" build graph """

# input placeholders
# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
y = tf.placeholder(tf.float32, shape=[None, mnist_data.NUM_LABELS], name='target_labels')

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
fack_id_in = tf.placeholder(tf.float32, shape=[None, mnist_data.NUM_LABELS], name='latent_variable')

# network architecture
x_, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x_hat, x, y, dim_img, dim_z, n_hidden, keep_prob)

decoded = vae.decoder(z_in, fack_id_in, dim_img, n_hidden)

sess = tf.InteractiveSession()

saver = tf.train.Saver()
saver = tf.train.import_meta_graph('models/mnist_gan.ckpt-'+model_no+'.meta')
saver.restore(sess, './models/mnist_gan.ckpt-'+model_no)

def OneHot(X, n=10, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

# indexes 1,11,21,31,... are ones, 2,12,22 are twos etc.
def generate_samples_for_digits(sample_size=100):
    
    Z_np_sample_buffer = np.random.randn(sample_size, dim_z)
    
    digits = np.zeros((sample_size,)).astype(int)
    for i in range(len(digits)):
        digits[i] = i%10
    Y_np_sample = OneHot( digits)

    generated_samples = sess.run(decoded, feed_dict={z_in: Z_np_sample_buffer, fack_id_in: Y_np_sample, keep_prob : 1})

    if (np.any(np.isnan(generated_samples))) or (not np.all(np.isfinite(generated_samples))):
        print('Problem')
        print(generate_samples_for_digits(10000)[0].max())
        print(generate_samples_for_digits(10000)[0].min())
        generated_samples = generate_samples_for_digits(sample_size)

    return generated_samples, digits

trX_vae, trY_vae = generate_samples_for_digits(1000000)
np.save('trX_vae', trX_vae)
np.save('trY_vae', trY_vae)