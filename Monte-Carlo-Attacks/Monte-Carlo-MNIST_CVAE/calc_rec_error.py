import tensorflow as tf
import numpy as np
import mnist_data
import vae

""" parameters """

model_no = '299'
# model_no = '249'

IMAGE_SIZE_MNIST = 28
n_hidden = 500
dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image
dim_z = 10

""" build graph """

# input placeholders
x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
y = tf.placeholder(tf.float32, shape=[None, mnist_data.NUM_LABELS], name='target_labels')

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# network architecture
rec_loss = vae.autoencoder_rec_loss(x, y, dim_img, dim_z, n_hidden, keep_prob)

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

def compute_avg_rec_error(x_sample, y_sample, repeats, n=3):
    y_sample = OneHot(y_sample)

    x_repeated = np.repeat([x_sample],repeats,axis=0)
    y_repeated =np.repeat(y_sample,repeats,axis=0)

    avg_dist = 0.0

    for i in range(n):
        avg_dist = avg_dist + sess.run(rec_loss, feed_dict={x: x_repeated, y: y_repeated, keep_prob : 1})

    return avg_dist/n