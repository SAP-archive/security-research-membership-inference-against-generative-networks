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
# /vis folder
# /data folder
# /models folder
# pip install pillow
# pip install scikit-image
# pip install scikit-learn

# to run:
# source activate tensorflow_p36 && python train_small.py 0.1 && python mc_attack_gan.py 500 5 && python mc_attack_gan.py 500 5 && python mc_attack_gan.py 500 5 && sudo shutdown -P now

# source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn
# source activate tensorflow_p36 && python CONSOLIDATED_MC_Attacks_MNIST.py 500 10 && sudo shutdown -P now

# source activate tensorflow_p36 && python train_small.py 0.1 && python CONSOLIDATED_MC_Attacks_MNIST.py 500 5 && python CONSOLIDATED_MC_Attacks_MNIST.py 500 5 && sudo shutdown -P now

# source activate tensorflow_p36 && python train_small.py 0.01 && python CONSOLIDATED_MC_Attacks_MNIST.py 500 15 && sudo shutdown -P now

# source activate tensorflow_p36 && python CONSOLIDATED_MC_Attacks_MNIST.py 500 10 && sudo shutdown -P now

#source activate tensorflow_p36 && python CONSOLIDATED_MC_Attacks_MNIST.py 500 6 && sudo shutdown -P now

# LOCAL:
# python CONSOLIDATED_MC_Attacks_MNIST.py 500 1
# AMI: 
# python MC_Attacks_MNIST.py 500 1

model_no = sys.argv[1] # which model to attack
exp_nos = int(sys.argv[2]) # how many different experiments ofr specific indexes

data_dir = 'data/'

instance_no = np.random.randint(10000)
experiment = 'CONSOLIDATED_MC_Attacks_MNIST' + str(instance_no)

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

saver = tf.train.import_meta_graph('models/mnist_gan.ckpt-'+model_no+'.meta')

saver.restore(sess, './models/mnist_gan.ckpt-'+model_no)

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


# random digits
def generate_samples_random_digits(sample_size=100):
    Z_np_sample = np.random.randn(sample_size, dim_z)
    Y_np_sample = OneHot( np.random.randint(10, size=[sample_size]))

    Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=sample_size)

    generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample:Z_np_sample,
                        Y_tf_sample:Y_np_sample
                        })
    generated_samples = (generated_samples + 1.)/2.
    return generated_samples

# indexes 1,11,21,31,... are ones, 2,12,22 are twos etc.
def generate_samples_for_digits(sample_size=100):
    
    Z_np_sample = np.random.randn(sample_size, dim_z)
    
    digits = np.zeros((sample_size,)).astype(int)
    for i in range(len(digits)):
        digits[i] = i%10
    Y_np_sample = OneHot( digits)

    Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=sample_size)

    generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        Z_tf_sample:Z_np_sample,
                        Y_tf_sample:Y_np_sample
                        })
    generated_samples = (generated_samples + 1.)/2.
    return generated_samples

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

    iterations = 1000
    results_attacks = np.zeros((iterations, ))

    for i in range(len(results_attacks)):
        np.random.shuffle(disc_results_train)
        results_attacks[i] = wb_attack_sample(disc_results_train[0:10], disc_results_validate)

    eleven_perc_wb_attack = results_attacks.mean()

    print('50_perc_wb_attack: %.3f'%(fifty_perc_wb_attack))
    print('11_perc_wb_attack: %.3f'%(eleven_perc_wb_attack))

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

        euclidean_generated_samples = generate_samples_random_digits(mc_sample_size)

        euclidean_generated_samples = euclidean_generated_samples - euclidean_generated_samples.min()
        euclidean_generated_samples = euclidean_generated_samples*255/euclidean_generated_samples.max()
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
        new_row['successfull_set_attack_1'] = mc_attack_results[6]
        new_row['successfull_set_attack_2'] = mc_attack_results[7]
        new_row['successfull_set_attack_3'] = mc_attack_results[8]
        
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

        euclidean_generated_samples = euclidean_generated_samples - euclidean_generated_samples.min()
        euclidean_generated_samples = euclidean_generated_samples*255/euclidean_generated_samples.max()
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
    new_row['method'] = 4 # euclidean PCA cat
    new_row['pca_n'] = n_components_pca
    new_row['percentage_of_data'] = percentage
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

def generate_batch_hog_features(samples):
    features_matrix = np.zeros((len(samples),81))

    for i in range(len(samples)):
        features_matrix[i] = hog(samples[i].reshape((28, 28)), orientations=9, pixels_per_cell=(9, 9), visualise=False) #, transform_sqrt=True, block_norm='L2-Hys')
    
    return features_matrix

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
    new_row['successfull_set_attack_1'] = mc_attack_results[6]
    new_row['successfull_set_attack_2'] = mc_attack_results[7]
    new_row['successfull_set_attack_3'] = mc_attack_results[8]
    
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))

def hog_mc_attack(trX_inds, vaX_inds, exp_no, mc_hog_no_batches, mc_sample_size, percentiles):

    feature_matrix_vaX = generate_batch_hog_features(vaX[vaX_inds])
    feature_matrix_trX = generate_batch_hog_features(trX[trX_inds])

    distances_trX = np.zeros((len(feature_matrix_trX), mc_hog_no_batches*mc_sample_size))
    distances_vaX = np.zeros((len(feature_matrix_vaX), mc_hog_no_batches*mc_sample_size))

    for i in range(mc_hog_no_batches):

        print('Working on %d/%d'%(i, mc_hog_no_batches))

        generated_samples = generate_samples_random_digits(mc_sample_size)

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
        new_row['successfull_set_attack_1'] = mc_attack_results[6]
        new_row['successfull_set_attack_2'] = mc_attack_results[7]
        new_row['successfull_set_attack_3'] = mc_attack_results[8]
        
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

    # white box attack
    wb_attack(trX_inds, vaX_inds, exp_no)
    print(experiment+': Finished White Box in experiment %d of %d'%(exp_no+1, exp_nos))
    
    ## hog mc attack 
    ## 100 iterations each having 10000 instances for monte carlo simulation
    ## higher amount of instances exceeds memory
    #hog_mc_attack(trX_inds, vaX_inds, exp_no, 100, 10000, [1,0.1,0.01, 0.001, 0.0001])
    #print(experiment+': Finished HOG (Default) Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))

    ## euclidean pca mc attack
    #euclidean_PCA_mc_attack(42, trX_inds, vaX_inds, exp_no, 100, 10000, [1,0.1,0.01,0.001,0.0001])
    #print(experiment+': Finished PCA Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))

    ## pca category
    ## 300 iterations each having 30000 instances for monte carlo simulation (1h together with below)
    euclidean_PCA_mc_attack_category(40, trX_inds, vaX_inds, exp_no, 333, 30000, [1,0.1, 0.01])
    print(experiment+': Finished CATEGORY PCA Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))

    # hog category
    hog_mc_attack_category(trX_inds, vaX_inds, exp_no, 100, 30000, [1,0.1, 0.01])
    print(experiment+': Finished CATEGORY HOG (Default) Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))
    
    print_elapsed_time()