import os
import numpy as np
from model import *
from load import load_mnist_with_valid_set
import time 
import scipy
import sys
from sklearn.decomposition import PCA
from skimage.feature import hog

n_epochs = 1000
learning_rate = 0.0002
batch_size = 128
image_shape = [28,28,1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1

# requirements
# /bb_models folder

# LOCAL:
# python bb_attack.py 500 1

model_no = sys.argv[1] # which model to attack
exp_nos = int(sys.argv[2]) # how many different experiments ofr specific indexes

data_dir = 'data/'

instance_no = np.random.randint(10000)
experiment = 'BB_Attack' + str(instance_no)

# only give the DCGAN 10% of training data
train_inds = np.loadtxt('train_inds.csv').astype(int)
percentage = np.loadtxt('percentage.csv')
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

saver = tf.train.import_meta_graph('bb_models/mnist_gan.ckpt-'+model_no+'.meta')

saver.restore(sess, './bb_models/mnist_gan.ckpt-'+model_no)

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

experiment_results = []

def OneHot(X, n=10, negative_class=0.):
    X = np.asarray(X).flatten()
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def print_elapsed_time():
    end_time = int(time.time())
    d = divmod(end_time-start_time,86400)  # days
    h = divmod(d[1],3600)  # hours
    m = divmod(h[1],60)  # minutes
    s = m[1]  # seconds

    print('Elapsed Time: %d days, %d hours, %d minutes, %d seconds' % (d[0],h[0],m[0],s))

def discriminate_for_wb(data_to_be_discriminated, labels_to_be_discriminated, training_indicator):
    disc_results = np.zeros((len(data_to_be_discriminated),2))
    
    data_to_be_discriminated = data_to_be_discriminated.reshape( [-1, 28, 28, 1]) / 255
    
    disc_results[:,1] = training_indicator
    
    for iteration in range(len(data_to_be_discriminated) // batch_size):
        start = iteration*batch_size
        end = (iteration+1)*batch_size
        
        ind = np.arange(start,end)
        Xs = tf.to_float(data_to_be_discriminated[ind].reshape( [-1, 28, 28, 1]) / 255)
        Ys = tf.to_float(OneHot(labels_to_be_discriminated[ind]))
        disc_results[ind, 0] = np.reshape(sess.run(dcgan_model.discriminate(Xs ,Ys)),(batch_size,))
    
    # fill last few elements
    ind = np.arange(len(data_to_be_discriminated)-batch_size,len(data_to_be_discriminated))
    Xs = tf.to_float(data_to_be_discriminated[ind].reshape( [-1, 28, 28, 1]) / 255)
    Ys = tf.to_float(OneHot(labels_to_be_discriminated[ind]))
    disc_results[ind, 0] = np.reshape(sess.run(dcgan_model.discriminate(Xs ,Ys)),(batch_size,))
    
    return disc_results

def wb_attack_sample(disc_results_train, disc_results_validate):
    results = np.concatenate((disc_results_train,disc_results_validate))
    np.random.shuffle(results)
    results = results[results[:,0].argsort()]

    return results[-len(disc_results_train):,1].mean()

def wb_attack(trX_inds, vaX_inds, exp_no):

    disc_results_train = discriminate_for_wb(trX[trX_inds],trY[trX_inds],1)
    disc_results_validate = discriminate_for_wb(vaX[vaX_inds],vaY[vaX_inds],0)

    fifty_perc_wb_attack = wb_attack_sample(disc_results_train, disc_results_validate)

    #iterations = 1000
    #results_attacks = np.zeros((iterations, ))

    #for i in range(len(results_attacks)):
    #    np.random.shuffle(disc_results_train)
    #    results_attacks[i] = wb_attack_sample(disc_results_train[0:10], disc_results_validate)

    eleven_perc_wb_attack = 0#results_attacks.mean()

    print('50_perc_wb_attack: %.3f'%(fifty_perc_wb_attack))
    #print('11_perc_wb_attack: %.3f'%(eleven_perc_wb_attack))

    # white box
    new_row = np.zeros(1, dtype = dt)[0]
    new_row['instance_no'] = instance_no
    new_row['exp_no'] = exp_no
    new_row['method'] = 1 # white box
    new_row['percentage_of_data'] = percentage
    new_row['50_perc_white_box'] = fifty_perc_wb_attack
    new_row['11_perc_white_box'] = eleven_perc_wb_attack
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

start_time = int(time.time())

for exp_no in range(exp_nos):

    trX_inds = np.arange(len(trX))
    np.random.shuffle(trX_inds)
    trX_inds = trX_inds[0:100]

    vaX_inds = np.arange(len(trX))
    np.random.shuffle(vaX_inds)
    vaX_inds = vaX_inds[0:100]

    # white box attack on shadow gan
    wb_attack(trX_inds, vaX_inds, exp_no)
    print(experiment+': Finished White Box in experiment %d of %d'%(exp_no+1, exp_nos))
    
    print_elapsed_time()