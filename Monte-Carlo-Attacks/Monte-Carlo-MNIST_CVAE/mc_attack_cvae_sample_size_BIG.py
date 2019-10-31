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

# combined:
# source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 300 && python mc_attack_cvae.py 299 5 && python mc_attack_cvae.py 299 5 && sudo shutdown -P now
# source activate tensorflow_p36 && python mc_attack_cvae_sample_size_BIG.py 299 5 && python mc_attack_cvae_sample_size_BIG.py 299 5 && python mc_attack_cvae_sample_size_BIG.py 299 5 && python mc_attack_cvae_sample_size_BIG.py 299 5 && sudo shutdown -P now

model_no = sys.argv[1] # which model to attack
exp_nos = int(sys.argv[2]) # how many different experiments ofr specific indexes

instance_no = np.random.randint(100000)
experiment = 'MC_ATTACK_CVAE' + str(instance_no)
percentage = np.loadtxt('percentage.csv')

dt = np.dtype([('instance_no', int),
               ('exp_no', int),
               ('method', int), # 1 = white box, 2 = euclidean_PCA, 3 = hog, 4 = euclidean_PCA category, 5 = hog category, 6 = ais
               ('pca_n', int),
               ('percentage_of_data', float),
               ('percentile', float),
               ('mc_euclidean_no_batches', int), # stuff
               ('50_perc_mc_attack_eps', float),
               ('mean_first_candidate', float),
               ('std_first_candidate', float),
               ('mean_last_candidate', float),
               ('std_last_candidate', float),
               ('mean_next_candidate', float),
               ('std_next_candidate', float),
               ('mean_abs_last_candidate', float),
               ('std_abs_last_candidate', float),
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

def f(x, d_min):
    if x<d_min:
        return 1
    else:
        return 0

f = np.vectorize(f)

def calculate_results(distances_trX, distances_vaX, d_min):

    distances_vaX_binarized = f(distances_vaX, d_min)
    distances_trX_binarized = f(distances_trX, d_min)

    results_sample = [[distances_vaX_binarized[i].mean(), distances_vaX_binarized[i].std()/np.sqrt(len(distances_vaX_binarized[i]))] for i in range(len(distances_vaX_binarized))]
    results_train = [[distances_trX_binarized[i].mean(), distances_trX_binarized[i].std()/np.sqrt(len(distances_trX_binarized[i]))] for i in range(len(distances_trX_binarized))]

    results_train_added_col = np.zeros((100, 3))
    results_train_added_col[:,2] = 1
    results_train_added_col[:,0:2] = np.reshape(results_train, (100,2))
    results_sample_added_col = np.zeros((100, 3))
    results_sample_added_col[:,0:2] = np.reshape(results_sample, (100,2))
    results = np.concatenate((results_sample_added_col, results_train_added_col))

    return results

def euclidean_PCA_mc_attack_category(n_components_pca, trX_inds, vaX_inds, exp_no, mc_euclidean_no_batches, mc_sample_size, percentiles):
    pca = PCA(n_components=n_components_pca)

    pca.fit_transform(teX.reshape((len(teX),784)))

    euclidean_trX = np.reshape(trX, (len(trX),784,))
    euclidean_trX = euclidean_trX[trX_inds]
    euclidean_trX = pca.transform(euclidean_trX)

    euclidean_vaX = np.reshape(vaX, (len(vaX),784,))
    euclidean_vaX = euclidean_vaX[vaX_inds]
    euclidean_vaX = pca.transform(euclidean_vaX)

    results = None

    for k in range(10):

        distances_trX = np.zeros((len(euclidean_trX), mc_euclidean_no_batches*mc_sample_size // 10))
        distances_vaX = np.zeros((len(euclidean_vaX), mc_euclidean_no_batches*mc_sample_size // 10))

        for i in range(mc_euclidean_no_batches):

            print('Working on %d/%d in iter %d/10'%(i, mc_euclidean_no_batches, k))

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

        distances = np.concatenate((distances_trX,distances_vaX))
        d_min = np.median([distances[i].min() for i in range(len(distances))])
        results_partial = calculate_results(distances_trX, distances_vaX, d_min)

        if results is None:
            results = results_partial
        else:
            results += results_partial

    print('Calculating Results Matrices for flexible d_min...')
   
    np.random.shuffle(results)
    results = results[results[:,0].argsort()]

    # save data
    new_row = np.zeros(1, dtype = dt)[0]
    new_row['instance_no'] = instance_no
    new_row['exp_no'] = exp_no
    new_row['method'] = 4 # euclidean PCA cat
    new_row['pca_n'] = n_components_pca
    new_row['percentage_of_data'] = percentage
    new_row['percentile'] = -1
    new_row['mc_euclidean_no_batches'] = mc_euclidean_no_batches

    new_row['50_perc_mc_attack_eps'] = results[-100:][:,2].mean()

    new_row['mean_first_candidate'] = results[-1][0]
    new_row['std_first_candidate'] = results[-1][1]
    new_row['mean_last_candidate'] = results[-100][0]
    new_row['std_last_candidate'] = results[-100][1]
    new_row['mean_next_candidate'] = results[-101][0]
    new_row['std_next_candidate'] = results[-101][1]
    new_row['mean_abs_last_candidate'] = results[-200][0]
    new_row['std_abs_last_candidate'] = results[-200][1]
        
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

    ## pca category
    # 8:00 mins 500
    # 500
    ## 300 iterations each having 30000 instances for monte carlo simulation (1h together with below)
    #[1, 3, 10, 30, 100, 300, 1000]
    for mc_no_samples in [300]:
        euclidean_PCA_mc_attack_category(40, trX_inds, vaX_inds, exp_no, mc_no_samples, 30000, [])
        print(experiment+': Finished CATEGORY PCA Monte Carlo in experiment %d of %d'%(exp_no+1, exp_nos))
    
    print_elapsed_time()