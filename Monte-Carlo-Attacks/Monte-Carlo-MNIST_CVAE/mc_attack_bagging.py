import tensorflow as tf
import numpy as np
import mnist_data
import os
import vae
import plot_utils
import glob
import sys
import time
import scipy
from sklearn.decomposition import PCA
from skimage.feature import hog

""" parameters """

# source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn
# source activate tensorflow_p36 && python run_main.py --dim_z 10 --num_epochs 300
# source activate tensorflow_p36 && python mc_attack_cvae.py 299 5 && python mc_attack_cvae.py 299 5 && sudo shutdown -P now

# combined:
# source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 300 && python mc_attack_cvae.py 299 5 && python mc_attack_cvae.py 299 5 && sudo shutdown -P now
# source activate tensorflow_p36 && python mc_attack_cvae.py 299 5 && python mc_attack_cvae.py 299 5 && sudo shutdown -P now
# source activate tensorflow_p36 && python mc_attack_cvae.py 299 5 && python mc_attack_cvae.py 299 5 && python mc_attack_cvae.py 299 5 && sudo shutdown -P now
model_no = sys.argv[1] # which model to attack
exp_nos = int(sys.argv[2]) # how many different experiments ofr specific indexes

instance_no = np.random.randint(10000)
experiment = 'MC_ATTACK_CVAE' + str(instance_no)
percentage = np.loadtxt('percentage.csv')

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
              ])

experiment_results = []

IMAGE_SIZE_MNIST = 28
n_hidden = 500
dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image
dim_z = 10

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
        print(generated_samples[0])
        print(generated_samples[1])
        generated_samples = generate_samples_for_digits(sample_size)

    return generated_samples

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

def euclidean_PCA_mc_attack(n_components_pca, trX_inds, vaX_inds, exp_no, mc_euclidean_no_batches, mc_sample_size, percentiles):
    pca = PCA(n_components=n_components_pca)

    pca.fit_transform(teX.reshape((len(teX),784)))

    euclidean_trX = np.reshape(trX, (len(trX),784,))
    euclidean_trX = euclidean_trX[trX_inds]
    euclidean_trX = pca.transform(euclidean_trX)

    euclidean_vaX = np.reshape(vaX, (len(vaX),784,))
    euclidean_vaX = euclidean_vaX[vaX_inds]
    euclidean_vaX = pca.transform(euclidean_vaX)

    distances_trX = np.zeros((len(euclidean_trX), mc_euclidean_no_batches*mc_sample_size))
    distances_vaX = np.zeros((len(euclidean_vaX), mc_euclidean_no_batches*mc_sample_size))

    for i in range(mc_euclidean_no_batches):

        print('Working on %d/%d'%(i, mc_euclidean_no_batches))

        euclidean_generated_samples = generate_samples_for_digits(mc_sample_size)

        #euclidean_generated_samples = euclidean_generated_samples - euclidean_generated_samples.min()
        #euclidean_generated_samples = euclidean_generated_samples*255/euclidean_generated_samples.max()
        euclidean_generated_samples = np.reshape(euclidean_generated_samples, (len(euclidean_generated_samples),784,))
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
        new_row['percentage_of_data'] = percentage
        new_row['percentile'] = percentile
        new_row['mc_euclidean_no_batches'] = mc_euclidean_no_batches

        mc_attack_results = mc_attack(results_sample, results_train)
        new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
        new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
        new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
        new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
        new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
        new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
        
        experiment_results.append(new_row)
        np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

def euclidean_PCA_mc_attack_category(n_components_pca, trX_inds, vaX_inds, exp_no, mc_euclidean_no_batches, mc_sample_size, percentiles):
    pca = PCA(n_components=n_components_pca)

    pca.fit_transform(teX.reshape((len(teX),784)))

    euclidean_trX = np.reshape(trX, (len(trX),784,))
    euclidean_trX = euclidean_trX[trX_inds]
    euclidean_trX = pca.transform(euclidean_trX)

    euclidean_vaX = np.reshape(vaX, (len(vaX),784,))
    euclidean_vaX = euclidean_vaX[vaX_inds]
    euclidean_vaX = pca.transform(euclidean_vaX)

    distances_trX = np.zeros((len(euclidean_trX), mc_euclidean_no_batches*mc_sample_size // 10))
    distances_vaX = np.zeros((len(euclidean_vaX), mc_euclidean_no_batches*mc_sample_size // 10))

    for i in range(mc_euclidean_no_batches):

        print('Working on %d/%d'%(i, mc_euclidean_no_batches))

        euclidean_generated_samples = generate_samples_for_digits(mc_sample_size)

        euclidean_generated_samples = np.reshape(euclidean_generated_samples, (len(euclidean_generated_samples),784,))
        euclidean_generated_samples = pca.transform(euclidean_generated_samples)
        
        for digit in range(10):
            # indexes of 1's, 2's, 3's etc.
            digit_indexes_train = np.where(trY[trX_inds] == digit)
            digit_indexes_sample = [digit+10*i for i in range(mc_sample_size//10)]
            # only compare to current digit
            distances_trX[digit_indexes_train,i*mc_sample_size//10:(i+1)*mc_sample_size//10] = scipy.spatial.distance.cdist(euclidean_trX[digit_indexes_train], euclidean_generated_samples[digit_indexes_sample], 'euclidean')

        for digit in range(10):
            # indexes of 1's, 2's, 3's etc.
            digit_indexes_va = np.where(vaY[vaX_inds] == digit)
            digit_indexes_sample = [digit+10*i for i in range(mc_sample_size//10)]
            # only compare to current digit
            distances_vaX[digit_indexes_va,i*mc_sample_size//10:(i+1)*mc_sample_size//10] = scipy.spatial.distance.cdist(euclidean_vaX[digit_indexes_va], euclidean_generated_samples[digit_indexes_sample], 'euclidean')
        
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
        new_row['method'] = 4 # euclidean PCA cat
        new_row['pca_n'] = n_components_pca
        new_row['percentage_of_data'] = percentage
        new_row['percentile'] = percentile
        new_row['mc_euclidean_no_batches'] = mc_euclidean_no_batches

        mc_attack_results = mc_attack(results_sample, results_train)
        new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
        new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
        new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
        new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
        new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
        new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
        
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
    new_row['method'] = 4 # euclidean PCA cat
    new_row['pca_n'] = n_components_pca
    new_row['percentage_of_data'] = percentage
    new_row['percentile'] = -1 # dynamic
    new_row['mc_euclidean_no_batches'] = mc_euclidean_no_batches
    mc_attack_results = mc_attack(results_sample, results_train)
    new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
    new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
    new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
    new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
    new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
    new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
    
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

    return results_sample,results_train

def generate_batch_hog_features(samples):
    features_matrix = np.zeros((len(samples),81))

    for i in range(len(samples)):
        features_matrix[i] = hog(samples[i].reshape((28, 28)), orientations=9, pixels_per_cell=(9, 9), visualise=False) #, transform_sqrt=True, block_norm='L2-Hys')
    
    return features_matrix

def hog_mc_attack(trX_inds, vaX_inds, exp_no, mc_hog_no_batches, mc_sample_size, percentiles):

    feature_matrix_vaX = generate_batch_hog_features(vaX[vaX_inds])
    feature_matrix_trX = generate_batch_hog_features(trX[trX_inds])

    distances_trX = np.zeros((len(feature_matrix_trX), mc_hog_no_batches*mc_sample_size))
    distances_vaX = np.zeros((len(feature_matrix_vaX), mc_hog_no_batches*mc_sample_size))

    for i in range(mc_hog_no_batches):

        print('Working on %d/%d'%(i, mc_hog_no_batches))

        generated_samples = generate_samples_for_digits(mc_sample_size)

        generated_samples = generated_samples - generated_samples.min()
        generated_samples = generated_samples*255/generated_samples.max()

        feature_matrix_generated = generate_batch_hog_features(generated_samples)

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
        new_row['method'] = 3
        new_row['percentage_of_data'] = percentage
        new_row['percentile'] = percentile
        new_row['mc_hog_no_batches'] = mc_hog_no_batches

        mc_attack_results = mc_attack(results_sample, results_train)
        new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
        new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
        new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
        new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
        new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
        new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
        
        experiment_results.append(new_row)
        np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

def hog_mc_attack_category(trX_inds, vaX_inds, exp_no, mc_hog_no_batches, mc_sample_size, percentiles):

    feature_matrix_vaX = generate_batch_hog_features(vaX[vaX_inds])
    feature_matrix_trX = generate_batch_hog_features(trX[trX_inds])

    distances_trX = np.zeros((len(feature_matrix_trX), mc_hog_no_batches*mc_sample_size // 10))
    distances_vaX = np.zeros((len(feature_matrix_vaX), mc_hog_no_batches*mc_sample_size // 10))

    for i in range(mc_hog_no_batches):

        print('Working on %d/%d'%(i, mc_hog_no_batches))

        generated_samples = generate_samples_for_digits(mc_sample_size)

        generated_samples = generated_samples - generated_samples.min()
        generated_samples = generated_samples*255/generated_samples.max()

        feature_matrix_generated = generate_batch_hog_features(generated_samples)

        for digit in range(10):
            # indexes of 1's, 2's, 3's etc.
            digit_indexes_train = np.where(trY[trX_inds] == digit)
            digit_indexes_sample = [digit+10*i for i in range(mc_sample_size//10)]
            # only compare to current digit
            distances_trX[digit_indexes_train,i*mc_sample_size//10:(i+1)*mc_sample_size//10] = scipy.spatial.distance.cdist(feature_matrix_trX[digit_indexes_train], feature_matrix_generated[digit_indexes_sample], 'euclidean')

        for digit in range(10):
            # indexes of 1's, 2's, 3's etc.
            digit_indexes_va = np.where(vaY[vaX_inds] == digit)
            digit_indexes_sample = [digit+10*i for i in range(mc_sample_size//10)]
            # only compare to current digit
            distances_vaX[digit_indexes_va,i*mc_sample_size//10:(i+1)*mc_sample_size//10] = scipy.spatial.distance.cdist(feature_matrix_vaX[digit_indexes_va], feature_matrix_generated[digit_indexes_sample], 'euclidean')

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
        new_row['method'] = 5 # hog cat
        new_row['percentage_of_data'] = percentage
        new_row['percentile'] = percentile
        new_row['mc_hog_no_batches'] = mc_hog_no_batches

        mc_attack_results = mc_attack(results_sample, results_train)
        new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
        new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
        new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
        new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
        new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
        new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
        
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
    new_row['method'] = 5 # hog cat
    new_row['percentage_of_data'] = percentage
    new_row['percentile'] = -1
    new_row['mc_hog_no_batches'] = mc_hog_no_batches

    mc_attack_results = mc_attack(results_sample, results_train)
    new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
    new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
    new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
    new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
    new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
    new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
    
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

    return results_sample,results_train

start_time = int(time.time())

for exp_no in range(exp_nos):

    trX_inds = np.arange(len(trX))
    np.random.shuffle(trX_inds)
    trX_inds = trX_inds[0:100]

    vaX_inds = np.arange(len(trX))
    np.random.shuffle(vaX_inds)
    vaX_inds = vaX_inds[0:100]

    # white box attack
    #wb_attack(trX_inds, vaX_inds, exp_no)
    #print(experiment+': Finished White Box in experiment %d of %d'%(exp_no+1, exp_nos))
    
    ## hog mc attack 
    ## 100 iterations each having 10000 instances for monte carlo simulation
    ## higher amount of instances exceeds memory
    # 100
    #hog_mc_attack(trX_inds, vaX_inds, exp_no, 100, 10000, [1,0.1,0.01, 0.001, 0.001])
    #print(experiment+': Finished HOG (Default) Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))

    ## euclidean pca mc attack
    ## 3 mins
    # 200
    #euclidean_PCA_mc_attack(40, trX_inds, vaX_inds, exp_no, 200, 10000, [1,0.1,0.01,0.001])
    #print(experiment+': Finished PCA Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))

    ## pca category
    # 8:00 mins 500
    # 500
    ## 300 iterations each having 30000 instances for monte carlo simulation (1h together with below)
    results_sample_pca,results_train_pca = euclidean_PCA_mc_attack_category(40, trX_inds, vaX_inds, exp_no, 300, 30000, [])
    print(experiment+': Finished CATEGORY PCA Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))

    # hog category (6s per Iteration, )
    # 300
    results_sample_hog,results_train_hog = hog_mc_attack_category(trX_inds, vaX_inds, exp_no, 150, 30000, [])
    print(experiment+': Finished CATEGORY HOG (Default) Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))

    pca_normalizer = np.max(np.concatenate((results_sample_pca[:,1],results_train_pca[:,1])))
    hog_normalizer = np.max(np.concatenate((results_sample_hog[:,1],results_train_hog[:,1])))
    results_sample_pca = results_sample_pca/pca_normalizer
    results_train_pca = results_train_pca/pca_normalizer
    results_sample_hog = results_sample_hog/hog_normalizer
    results_train_hog = results_train_hog/hog_normalizer

    results_sample_combined = results_sample_pca+results_sample_hog
    results_train_combined = results_train_pca+results_train_hog
    results_train_combined[:,0]=1

    # save data
    new_row = np.zeros(1, dtype = dt)[0]
    new_row['instance_no'] = instance_no
    new_row['exp_no'] = exp_no
    new_row['method'] = 9 # bagging
    new_row['percentage_of_data'] = percentage
    new_row['percentile'] = -1
    new_row['mc_hog_no_batches'] = 0

    mc_attack_results = mc_attack(results_sample_combined, results_train_combined)
    new_row['50_perc_mc_attack_log'] = mc_attack_results[0]
    new_row['50_perc_mc_attack_eps'] = mc_attack_results[1]
    new_row['50_perc_mc_attack_frac'] = mc_attack_results[2]
    new_row['11_perc_mc_attack_log'] = mc_attack_results[3]
    new_row['11_perc_mc_attack_eps'] = mc_attack_results[4]
    new_row['11_perc_mc_attack_frac'] = mc_attack_results[5]
    
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))
    
    print_elapsed_time()