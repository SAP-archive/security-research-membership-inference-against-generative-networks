import os

from GAN import GAN
from utils import show_all_variables

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
import time
import sys
import scipy
from sklearn.decomposition import PCA

# python mc_attack.py 5 && python mc_attack.py 5 && python mc_attack.py 5 && python mc_attack.py 5 && sudo shutdown -P now

exp_nos = int(sys.argv[1]) # how many different experiments ofr specific indexes
instance_no = np.random.randint(10000)
experiment = 'CIFAR10_MC_ATTACK' + str(instance_no)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
gan = GAN(sess, epoch=3500, batch_size=100, dataset_name='cifar10',
                      checkpoint_dir='checkpoint', result_dir='results', log_dir='logs', directory='./train', reuse=True)
gan.build_model()
gan.load_model()

bs = 100
image_dims = [32, 32, 3]
inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='sample_images')

D_real, D_real_logits, _ = gan.discriminator(inputs, is_training=True, reuse=True)

trX, trY, vaX, vaY = load_cifar10_with_validation('./train', reuse=True)
teX = vaX[44000:]
teY = vaY[44000:]
vaX = vaX[:44000]
vaY = vaY[:44000]

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

def print_elapsed_time():
    end_time = int(time.time())
    d = divmod(end_time-start_time,86400)  # days
    h = divmod(d[1],3600)  # hours
    m = divmod(h[1],60)  # minutes
    s = m[1]  # seconds

    print('Elapsed Time: %d days, %d hours, %d minutes, %d seconds' % (d[0],h[0],m[0],s))

def calculate_results_matrices(distances_real_vs_sample,distances_real_vs_train, d_min=0.1):

    results_sample = np.zeros((len(distances_real_vs_sample),4))
    for i in range(len(results_sample)):
        # indicate that dataset is a sample
        results_sample[i][0] = 0
        
        integral_approx = 0
        integral_approx_log = 0
        integral_approx_eps = 0
        for eps in distances_real_vs_sample[i]:
            if eps < d_min:
                integral_approx = integral_approx + d_min/eps
                integral_approx_log = integral_approx_log + (-np.log(eps/d_min))
                integral_approx_eps = integral_approx_eps + 1

        integral_approx = integral_approx/len(distances_real_vs_sample[0])
        integral_approx_log = integral_approx_log/len(distances_real_vs_sample[0])
        integral_approx_eps = integral_approx_eps/len(distances_real_vs_sample[0])

        results_sample[i][1] = integral_approx_log
        results_sample[i][2] = integral_approx_eps
        results_sample[i][3] = integral_approx

    results_train = np.zeros((len(distances_real_vs_train),4))
    for i in range(len(results_train)):
        # indicate that dataset is a training data set
        results_train[i][0] = 1
        
        integral_approx = 0
        integral_approx_log = 0
        integral_approx_eps = 0
        for eps in distances_real_vs_train[i]:
            if eps < d_min:
                integral_approx = integral_approx + d_min/eps
                integral_approx_log = integral_approx_log + (-np.log(eps/d_min))
                integral_approx_eps = integral_approx_eps + 1

        integral_approx = integral_approx/len(distances_real_vs_train[0])
        integral_approx_log = integral_approx_log/len(distances_real_vs_train[0])
        integral_approx_eps = integral_approx_eps/len(distances_real_vs_train[0])

        results_train[i][1] = integral_approx_log
        results_train[i][2] = integral_approx_eps
        results_train[i][3] = integral_approx
        
    return results_sample,results_train

def mc_attack_sample(results_sample, results_train):
    results = np.concatenate((results_sample, results_train))
    np.random.shuffle(results)
    mc_attack_log = results[results[:,1].argsort()][:,0][-len(results_train):].mean()
    np.random.shuffle(results)
    mc_attack_eps = results[results[:,2].argsort()][:,0][-len(results_train):].mean()
    np.random.shuffle(results)
    mc_attack_frac = results[results[:,3].argsort()][:,0][-len(results_train):].mean()

    successfull_set_attack_1 = results_train[:,1].sum() > results_sample[:,1].sum()
    successfull_set_attack_2 = results_train[:,2].sum() > results_sample[:,2].sum()
    successfull_set_attack_3 = results_train[:,3].sum() > results_sample[:,3].sum()

    return mc_attack_log, mc_attack_eps, mc_attack_frac, successfull_set_attack_1, successfull_set_attack_2, successfull_set_attack_3

def mc_attack(results_sample, results_train):

    mc_attack_log, mc_attack_eps, mc_attack_frac, successfull_set_attack_1, successfull_set_attack_2, successfull_set_attack_3 = mc_attack_sample(results_sample, results_train)

    print('50_perc_mc_attack_log: %.3f'%(mc_attack_log))
    print('50_perc_mc_attack_eps: %.3f'%(mc_attack_eps))
    print('50_perc_mc_attack_frac: %.3f'%(mc_attack_frac))
    print('successfull_set_attack_1: %.3f'%(successfull_set_attack_1))
    print('successfull_set_attack_2: %.3f'%(successfull_set_attack_2))
    print('successfull_set_attack_3: %.3f'%(successfull_set_attack_3))

    iterations = 1000
    results_attacks = np.zeros((iterations, 3))

    for i in range(len(results_attacks)):
        np.random.shuffle(results_train)
        res = mc_attack_sample(results_sample, results_train[0:10])
        results_attacks[i][0] = res[0]
        results_attacks[i][1] = res[1]
        results_attacks[i][2] = res[2]

    print('11_perc_mc_attack_log: %.3f'%(results_attacks[:,0].mean()))
    print('11_perc_mc_attack_eps: %.3f'%(results_attacks[:,1].mean()))
    print('11_perc_mc_attack_frac: %.3f'%(results_attacks[:,2].mean()))

    return mc_attack_log, mc_attack_eps, mc_attack_frac, results_attacks[:,0].mean(), results_attacks[:,1].mean(), results_attacks[:,2].mean(), successfull_set_attack_1, successfull_set_attack_2, successfull_set_attack_3

def calc_hist(image):
    vMin = np.amin(image)
    vMax = np.amax(image)

    image = (image-vMin)/(vMax-vMin)*255
    hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist,hist).flatten()
    return hist

def calc_batch_hist(images):
    features = np.zeros((len(images),4096))
    
    for i in range(len(images)):
        features[i,:] = calc_hist(images[i])
        
    return features

def color_hist_attack(mc_no_batches, mc_sample_size, trX_inds, vaX_inds, exp_no, percentiles):

    feature_matrix_vaX = calc_batch_hist(vaX[vaX_inds])
    feature_matrix_trX = calc_batch_hist(trX[trX_inds])

    distances_trX = np.zeros((len(feature_matrix_trX), mc_no_batches*mc_sample_size))
    distances_vaX = np.zeros((len(feature_matrix_vaX), mc_no_batches*mc_sample_size))

    for i in range(mc_no_batches):

        print('Working on %d/%d'%(i, mc_no_batches))

        generated_samples = gan.sample()

        feature_matrix_generated = calc_batch_hist(generated_samples)

        distances_trX_partial = scipy.spatial.distance.cdist(feature_matrix_trX, feature_matrix_generated, 'euclidean')
        distances_vaX_partial = scipy.spatial.distance.cdist(feature_matrix_vaX, feature_matrix_generated, 'euclidean')

        # optimized, better than concatenate
        distances_trX[:,i*mc_sample_size:(i+1)*mc_sample_size] = distances_trX_partial
        distances_vaX[:,i*mc_sample_size:(i+1)*mc_sample_size] = distances_vaX_partial

        print_elapsed_time()

    for percentile in percentiles:
        print_elapsed_time()
        print('Calculating Results Matrices for '+str(percentile)+' Percentile...')
        d_min = np.percentile(np.concatenate((distances_trX,distances_vaX)),percentile)
        results_sample,results_train = calculate_results_matrices(distances_vaX, distances_trX,d_min)
        # save data
        new_row = np.zeros(1, dtype = dt)[0]
        new_row['instance_no'] = instance_no
        new_row['exp_no'] = exp_no
        new_row['method'] = 8
        new_row['percentage_of_data'] = 0.1
        new_row['percentile'] = percentile
        new_row['mc_euclidean_no_batches'] = mc_no_batches
        mc_attack_results = mc_attack(results_sample, results_train)
        new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
        new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
        new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
        new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
        new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
        new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
        new_row['successfull_set_attack_1'] = mc_attack_results[6]
        new_row['successfull_set_attack_2'] = mc_attack_results[7]
        new_row['successfull_set_attack_3'] = mc_attack_results[8]
        experiment_results.append(new_row)
        np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

    print('Calculating Results Matrices for flexible d_min...')
    distances = np.concatenate((distances_trX,distances_vaX))
    d_min = np.median([distances[i].min() for i in range(len(distances))])
    results_sample,results_train = calculate_results_matrices(distances_vaX, distances_trX,d_min)

    # save data
    new_row = np.zeros(1, dtype = dt)[0]
    new_row['instance_no'] = instance_no
    new_row['exp_no'] = exp_no
    new_row['method'] = 8
    new_row['percentage_of_data'] = 0.1
    new_row['percentile'] = -1
    new_row['mc_euclidean_no_batches'] = mc_no_batches
    mc_attack_results = mc_attack(results_sample, results_train)
    new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
    new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
    new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
    new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
    new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
    new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
    new_row['successfull_set_attack_1'] = mc_attack_results[6]
    new_row['successfull_set_attack_2'] = mc_attack_results[7]
    new_row['successfull_set_attack_3'] = mc_attack_results[8]
    
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

def euclidean_PCA_mc_attack(n_components_pca, trX_inds, vaX_inds, exp_no, mc_euclidean_no_batches, mc_sample_size, percentiles):
    pca = PCA(n_components=n_components_pca)

    pca.fit_transform(teX.reshape((len(teX),3072)))

    euclidean_trX = np.reshape(trX, (len(trX),3072))
    euclidean_trX = euclidean_trX[trX_inds]
    euclidean_trX = pca.transform(euclidean_trX)

    euclidean_vaX = np.reshape(vaX, (len(vaX),3072))
    euclidean_vaX = euclidean_vaX[vaX_inds]
    euclidean_vaX = pca.transform(euclidean_vaX)

    distances_trX = np.zeros((len(euclidean_trX), mc_euclidean_no_batches*mc_sample_size))
    distances_vaX = np.zeros((len(euclidean_vaX), mc_euclidean_no_batches*mc_sample_size))

    for i in range(mc_euclidean_no_batches):

        print('Working on %d/%d'%(i, mc_euclidean_no_batches))

        euclidean_generated_samples = gan.sample()

        euclidean_generated_samples = np.reshape(euclidean_generated_samples, (len(euclidean_generated_samples),3072))
        euclidean_generated_samples = pca.transform(euclidean_generated_samples)

        distances_trX_partial = scipy.spatial.distance.cdist(euclidean_trX, euclidean_generated_samples, 'euclidean')
        distances_vaX_partial = scipy.spatial.distance.cdist(euclidean_vaX, euclidean_generated_samples, 'euclidean')

        # optimized, better than concatenate
        distances_trX[:,i*mc_sample_size:(i+1)*mc_sample_size] = distances_trX_partial
        distances_vaX[:,i*mc_sample_size:(i+1)*mc_sample_size] = distances_vaX_partial
        
        print_elapsed_time()

    for percentile in percentiles:
        print_elapsed_time()
        print('Calculating Results Matrices for '+str(percentile)+' Percentile...')

        d_min = np.percentile(np.concatenate((distances_trX,distances_vaX)),percentile)
        results_sample,results_train = calculate_results_matrices(distances_vaX, distances_trX,d_min)
        
        # save data
        new_row = np.zeros(1, dtype = dt)[0]
        new_row['instance_no'] = instance_no
        new_row['exp_no'] = exp_no
        new_row['method'] = 2 # euclidean PCA
        new_row['pca_n'] = n_components_pca
        new_row['percentage_of_data'] = 0.1
        new_row['percentile'] = percentile
        new_row['mc_euclidean_no_batches'] = mc_euclidean_no_batches

        mc_attack_results = mc_attack(results_sample, results_train)
        new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
        new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
        new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
        new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
        new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
        new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
        new_row['successfull_set_attack_1'] = mc_attack_results[6]
        new_row['successfull_set_attack_2'] = mc_attack_results[7]
        new_row['successfull_set_attack_3'] = mc_attack_results[8]
        
        experiment_results.append(new_row)
        np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

    print('Calculating Results Matrices for flexible d_min...')
    distances = np.concatenate((distances_trX,distances_vaX))
    d_min = np.median([distances[i].min() for i in range(len(distances))])
    results_sample,results_train = calculate_results_matrices(distances_vaX, distances_trX,d_min)

    # save data
    new_row = np.zeros(1, dtype = dt)[0]
    new_row['instance_no'] = instance_no
    new_row['exp_no'] = exp_no
    new_row['method'] = 2 # euclidean PCA
    new_row['pca_n'] = n_components_pca
    new_row['percentage_of_data'] = 0.1
    new_row['percentile'] = -1
    new_row['mc_euclidean_no_batches'] = mc_euclidean_no_batches

    mc_attack_results = mc_attack(results_sample, results_train)
    new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
    new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
    new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
    new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
    new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
    new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
    new_row['successfull_set_attack_1'] = mc_attack_results[6]
    new_row['successfull_set_attack_2'] = mc_attack_results[7]
    new_row['successfull_set_attack_3'] = mc_attack_results[8]
    
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

def discriminate_for_wb(data_to_be_discriminated, training_indicator):
    disc_results = np.zeros((len(data_to_be_discriminated),2))
    
    disc_results[:,1] = training_indicator
    disc_results[:,0] = sess.run(D_real, feed_dict={inputs:data_to_be_discriminated}).reshape((100,))
    
    return disc_results

def wb_attack_sample(disc_results_train, disc_results_validate):
    results = np.concatenate((disc_results_train,disc_results_validate))
    np.random.shuffle(results)
    results = results[results[:,0].argsort()]

    return results[-len(disc_results_train):,1].mean()

def wb_attack(trX_inds, vaX_inds, exp_no):

    disc_results_train = discriminate_for_wb(trX[trX_inds],1)
    disc_results_validate = discriminate_for_wb(vaX[vaX_inds],0)

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
    new_row['percentage_of_data'] = 0.1
    new_row['50_perc_white_box'] = fifty_perc_wb_attack
    new_row['11_perc_white_box'] = eleven_perc_wb_attack
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))
    
    #return fifty_perc_wb_attack

start_time = int(time.time())

for exp_no in range(exp_nos):

    trX_inds = np.arange(len(trX))
    np.random.shuffle(trX_inds)
    trX_inds = trX_inds[0:100]

    vaX_inds = np.arange(len(vaX))
    np.random.shuffle(vaX_inds)
    vaX_inds = vaX_inds[0:100]

    # euclidean PCA
    #euclidean_PCA_mc_attack(200, trX_inds, vaX_inds, exp_no, 100, 10000, [1,0.1,0.01,0.001])
    #print(experiment+': Finished PCA Monte Carlo 200 in experiment %d of %d'%(exp_no+1, exp_nos))

    euclidean_PCA_mc_attack(120, trX_inds, vaX_inds, exp_no, 300, 10000, [1,0.1,0.01,0.001])
    print(experiment+': Finished PCA Monte Carlo 120 in experiment %d of %d'%(exp_no+1, exp_nos))

    #euclidean_PCA_mc_attack(40, trX_inds, vaX_inds, exp_no, 100, 10000, [1,0.1,0.01,0.001])
    #print(experiment+': Finished PCA Monte Carlo 40 in experiment %d of %d'%(exp_no+1, exp_nos))

    # color_hist_attack
    # 10000 cannot be changed easily!
    color_hist_attack(300, 10000, trX_inds, vaX_inds, exp_no, [1, 0.1, 0.01, 0.001])
    print(experiment+': Finished Color Hist in experiment %d of %d'%(exp_no+1, exp_nos))

    # white box
    wb_attack(trX_inds, vaX_inds, exp_no)
    print(experiment+': Finished White Box in experiment %d of %d'%(exp_no+1, exp_nos))
    
    print_elapsed_time()