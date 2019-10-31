import os
import numpy as np
from model import *
from util import *
from load import load_mnist_with_valid_set
import time 

import tensorflow as tf
import numpy as np
import ais
import matplotlib.pyplot as plt
from priors import NormalPrior
from kernels import ParsenDensityEstimator
from scipy.stats import norm

#source activate tensorflow_p36; for i in $(seq 0 15); do echo $i; python ais_attack.py; done; sudo shutdown -P now

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

experiment_results = np.array([], dtype=dt)
try:
    experiment_results = np.loadtxt('AIS_SIGMA_0.025_.csv', dtype=dt)
    print('REUSE')
except:
    print('NO REUSE')

instance_no = np.random.randint(1000)
experiment = 'AIS_SIGMA_0.025_'+str(instance_no)

n_epochs = 1000
learning_rate = 0.0002
batch_size = 128
image_shape = [28,28,1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1

visualize_dim=196

data_dir = 'data/'

# only give the DCGAN 10% of training data
percentage = np.loadtxt('percentage.csv')
train_inds = np.loadtxt('train_inds.csv').astype(int)
trX, vaX, teX, trY, vaY, teY = load_mnist_with_valid_set(train_inds, percentage=percentage, data_dir=data_dir)

dcgan_model = DCGAN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        )

Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()

saver = tf.train.Saver()

saver = tf.train.import_meta_graph('models/mnist_gan.ckpt-500.meta')

saver.restore(sess, './models/mnist_gan.ckpt-500')

def save_image(data, path):
    img = np.zeros((28, 28, 3))
    img[0:28, 0:28, :] = data.reshape( [28, 28, 1])
    scipy.misc.imsave('./'+path+'.jpg', img)

def print_elapsed_time():
    end_time = int(time.time())
    d = divmod(end_time-start_time,86400)  # days
    h = divmod(d[1],3600)  # hours
    m = divmod(h[1],60)  # minutes
    s = m[1]  # seconds

    print('Elapsed Time: %d days, %d hours, %d minutes, %d seconds' % (d[0],h[0],m[0],s))

start_time = int(time.time())

class Generator(object):
    def __init__(self, digit):
        self.input_dim = dim_z
        self.output_dim = 784
        self.digit = digit

    def __call__(self, z):
        return self.generate_samples_tf(z, self.digit)

    def generate_samples_tf(self, Z,digit):
        buffer_sample_size = 200
        
        sample_size = tf.shape(Z)[0]

        #Z_np_sample_buffer = np.random.uniform(-1, 1, size=(buffer_sample_size,dim_z))
        Z_np_sample_buffer = np.random.randn(buffer_sample_size, dim_z)
        Z_np_sample = tf.concat([Z, Z_np_sample_buffer], 0)
        
        one_hot_digits = np.zeros(10,)
        one_hot_digits[digit] = 1
        Y_np_sample = tf.reshape(tf.tile(tf.constant(one_hot_digits, dtype=tf.float32),[sample_size]), [sample_size, 10])
        
        Y_np_sample_buffer = OneHot( np.random.randint(10, size=[buffer_sample_size]))
        Y_np_sample = tf.concat([Y_np_sample,Y_np_sample_buffer], 0)
        
        Z_np_sample = tf.cast(Z_np_sample, tf.float32)
        Y_np_sample = tf.cast(Y_np_sample, tf.float32)

        generated_samples = dcgan_model.samples_generator_var_batch(Z_np_sample, Y_np_sample)
        
        # transform to match validation data
        generated_samples = generated_samples - tf.reduce_min(generated_samples, axis=None)
        generated_samples = generated_samples*1/tf.reduce_max(generated_samples, axis=None)
        generated_samples = tf.reshape(generated_samples, [tf.shape(generated_samples)[0], 784])
        
        # remove buffer
        generated_samples = tf.slice(generated_samples, [0, 0], [sample_size, 784])

        # cast to int
        #generated_samples = tf.cast(generated_samples, tf.int32)

        return generated_samples


#index = 10
#digit = trY[index]
#save_image(trX[index], 'attack_it')

def mc_attack_sample(results_sample, results_train):
    results = np.concatenate((results_sample, results_train))
    np.random.shuffle(results)
    mc_attack_log = results[results[:,1].argsort()][:,0][-len(results_train):].mean()
    #print(results[results[:,1].argsort()])
    np.random.shuffle(results)
    mc_attack_eps = results[results[:,2].argsort()][:,0][-len(results_train):].mean()
    #print(results[results[:,2].argsort()])
    np.random.shuffle(results)
    mc_attack_frac = results[results[:,3].argsort()][:,0][-len(results_train):].mean()
    #print(results[results[:,3].argsort()])

    successfull_set_attack_1 = results_train[:,1].sum() > results_sample[:,1].sum()
    successfull_set_attack_2 = results_train[:,2].sum() > results_sample[:,2].sum()
    successfull_set_attack_3 = results_train[:,3].sum() > results_sample[:,3].sum()

    return mc_attack_log, mc_attack_eps, mc_attack_frac, successfull_set_attack_1, successfull_set_attack_2, successfull_set_attack_3

def ais_attack(trX_inds, vaX_inds, sigma, exp_no, instance_no):

    results_train = np.zeros((len(trX_inds),4))
    for i in range(len(trX_inds)):
        print('TRAIN ATTACK %d'%(i))
        print_elapsed_time()
        ais_val = ais_value(trY[trX_inds[i]], trX[trX_inds[i]], sigma)
        #print('###ais_val %.3f'%(ais_val))
        results_train[i,0] = 1
        results_train[i,1] = ais_val[0]
        results_train[i,2] = ais_val[1]
        results_train[i,3] = ais_val[2]

    results_sample = np.zeros((len(vaX_inds),4))
    for i in range(len(vaX_inds)):
        print('SAMPLE ATTACK %d'%(i))
        print_elapsed_time()
        ais_val = ais_value(vaY[vaX_inds[i]], vaX[vaX_inds[i]], sigma)
        #print('###ais_val %.3f'%(ais_val))
        results_sample[i,0] = 0
        results_sample[i,1] = ais_val[0]
        results_sample[i,2] = ais_val[1]
        results_sample[i,3] = ais_val[2]

    # save data
    new_row = np.zeros(1, dtype = dt)
    new_row[0]['instance_no'] = instance_no
    new_row[0]['exp_no'] = exp_no
    new_row[0]['method'] = 6
    new_row[0]['percentage_of_data'] = percentage
    new_row[0]['sigma_ais'] = sigma

    mc_attack_results = mc_attack_sample(results_sample, results_train)
    new_row[0]['50_perc_ais_acc_rate'] = mc_attack_results[0]
    new_row[0]['50_perc_ais'] = mc_attack_results[1]
    new_row['successfull_set_attack_1'] = mc_attack_results[3]
    new_row['successfull_set_attack_2'] = mc_attack_results[4]
    new_row['successfull_set_attack_3'] = mc_attack_results[5]
    
    try:
        np.savetxt(experiment+'.csv', np.concatenate((experiment_results,new_row)))
    except:
        np.savetxt(experiment+'.csv', np.concatenate(([experiment_results],new_row)))


def ais_value(digit, val, sigma):
    generator = Generator(digit)
    prior = NormalPrior()
    kernel = ParsenDensityEstimator()
    # 0.025
    # or 0.25
    model = ais.Model(generator, prior, kernel, sigma, 10000)

    num_samples = 1
    schedule = ais.get_schedule(num_samples, rad=4)
    p2 = model.ais(np.array([val/255]), schedule)

    return p2

exp_nos = 1

for exp_no in range(exp_nos):

    trX_inds = np.arange(len(trX))
    np.random.shuffle(trX_inds)
    trX_inds = trX_inds[0:50] # or 100?

    vaX_inds = np.arange(len(trX))
    np.random.shuffle(vaX_inds)
    vaX_inds = vaX_inds[0:50] # or 100?

    # ais attack
    ais_attack(trX_inds, vaX_inds, 0.025, 1, instance_no)
    print('### '+ experiment+': Finished AIS in experiment %d of %d'%(exp_no+1, exp_nos))
    
    print_elapsed_time()