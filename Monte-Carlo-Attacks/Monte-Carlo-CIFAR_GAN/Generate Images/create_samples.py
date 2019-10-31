import os

from GAN import GAN
from utils import show_all_variables
from utils import load_cifar10_with_validation

import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import load_cifar10_with_validation
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
gan = GAN(sess, epoch=3500, batch_size=100, dataset_name='cifar10', checkpoint_dir='checkpoint', result_dir='results', log_dir='logs', directory='./train', reuse=True)
gan.build_model()
gan.load_model()

sample_iters = 100

sample_matrix = np.zeros((10000*sample_iters, 32, 32, 3))

for i in range(sample_iters):
    print(i)
    samples = gan.sample()
    sample_matrix[i*10000:(i+1)*10000] = samples

np.save('sample_matrix',sample_matrix)