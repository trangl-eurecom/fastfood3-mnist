## Copyright 2016 Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes


from dataset import DataSet
import utils
import likelihoods
from dgp_rff import DgpRff

# import baselines

import losses

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
ops.NotDifferentiable("FastFood2")

def process_mnist(images, dtype = dtypes.float32, reshape=True):
    if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    return images

def get_data_info(images):
    rows, cols = images.shape
    std = np.zeros(cols)
    mean = np.zeros(cols)
    for col in range(cols):
        std[col] = np.std(images[:,col])
        mean[col] = np.mean(images[:,col])
    return mean, std

def standardize_data(images, means, stds):
    data = images.copy()
    rows, cols = data.shape
    for col in range(cols):
        if stds[col] == 0:
            data[:,col] = (data[:,col] - means[col])
        else:
            data[:,col] = (data[:,col] - means[col]) / stds[col]
    return data

def import_mnist():
    """
    This import mnist and saves the data as an object of our DataSet class
    :return:
    """
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 0
    ONE_HOT = True
    TRAIN_DIR = 'MNIST_data'


    local_file = base.maybe_download(TRAIN_IMAGES, TRAIN_DIR,
                                     SOURCE_URL + TRAIN_IMAGES)
    train_images = extract_images(open(local_file))

    local_file = base.maybe_download(TRAIN_LABELS, TRAIN_DIR,
                                     SOURCE_URL + TRAIN_LABELS)
    train_labels = extract_labels(open(local_file), one_hot=ONE_HOT)

    local_file = base.maybe_download(TEST_IMAGES, TRAIN_DIR,
                                     SOURCE_URL + TEST_IMAGES)
    test_images = extract_images(open(local_file))

    local_file = base.maybe_download(TEST_LABELS, TRAIN_DIR,
                                     SOURCE_URL + TEST_LABELS)
    test_labels = extract_labels(open(local_file), one_hot=ONE_HOT)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    ## Process images
    train_images = process_mnist(train_images)
    validation_images = process_mnist(validation_images)
    test_images = process_mnist(test_images)

    ## Standardize data
    train_mean, train_std = get_data_info(train_images)
#    train_images = standardize_data(train_images, train_mean, train_std)
#    validation_images = standardize_data(validation_images, train_mean, train_std)
#    test_images = standardize_data(test_images, train_mean, train_std)

    data = DataSet(train_images, train_labels)
    test = DataSet(test_images, test_labels)
    val = DataSet(validation_images, validation_labels)

    return data, test, val

# This function will create a train set (a subset of data) such that the number of each class are the same (if possible)
def extract_balance_train_set(data, data_size, train_size, nb_class):
    all_train_set = data.next_batch(data_size)
    allX = all_train_set[0]
    allY = all_train_set[1]
    nbEachClass = int(train_size / nb_class)
    cur_nbEachClass = np.zeros([nb_class])
    trainX = []
    trainY = []
    ind = 0
    while (len(trainX) < train_size) and (ind < data_size):
        class_nb = np.multiply(allY[ind], np.arange(nb_class))
        class_nb = int(np.sum(class_nb))
        if (cur_nbEachClass[class_nb] < nbEachClass):
            trainX.append(allX[ind])
            trainY.append(allY[ind])
            cur_nbEachClass[class_nb] = cur_nbEachClass[class_nb] + 1
        ind = ind + 1
        
    if (ind >= data_size):
        trainX = allX[0:train_size]
        trainY = allY[0:train_size]
        
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY
    
if __name__ == '__main__':
    
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    #tf.set_random_seed(FLAGS.seed)
    #np.random.seed(FLAGS.seed)

    data, test, _ = import_mnist()
    
    trainX, trainY = extract_balance_train_set(data, 55000, FLAGS.train_size, 10)
    testX_testY = test.next_batch(10000)
    testX = testX_testY[0]
    testY = testX_testY[1]
    
    
    ## Here we define a custom loss for dgp to show
    error_rate = losses.ZeroOneLoss(data.Dout)

    ## Likelihood
    like = likelihoods.Softmax()

    ## Optimizer
    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    ## Main dgp object
    #dgp = DgpRff(like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree, FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, FLAGS.learn_Omega)
    dgp = DgpRff(like, len(trainX), len(trainX[0]), len(trainY[0]), FLAGS.nl, FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree, FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, FLAGS.learn_Omega)
    
    print("Learning phase")
    ## Learning
    dgp.learn(trainX, trainY, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
              FLAGS.display_step, testX, testY, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints)
gialac@gateway-ssh:~/fastfood2-mnist$ cat dgp_rff.py
## Copyright 2016 Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from __future__ import print_function

import tensorflow as tf
import numpy as np
from dataset import DataSet
import utils
import likelihoods
import time
import os
import random_vector

current_milli_time = lambda: int(round(time.time() * 1000))

class DgpRff(object):
    def __init__(self, likelihood_fun, num_examples, d_raw_in, d_out, n_layers, n_rff, df, kernel_type, kernel_arccosine_degree, is_ard, feed_forward, q_Omega_fixed, theta_fixed, learn_Omega):
        """
        :param likelihood_fun: Likelihood function
        :param num_examples: total number of input samples
        :param d_raw_in: Dimensionality of the input, i.e for MNIST d_raw_in = 28 * 28 * 1, d_raw_in = 32 * 32 * 3
        :param d_out: Dimensionality of the output
        :param n_layers: Number of hidden layers
        :param n_rff: Number of random features for each layer
        :param df: Number of GPs for each layer
        :param kernel_type: Kernel type: currently only random Fourier features for RBF and arccosine kernels are implemented
        :param kernel_arccosine_degree: degree parameter of the arccosine kernel
        :param is_ard: Whether the kernel is ARD or isotropic
        :param feed_forward: Whether the original inputs should be fed forward as input to each layer
        :param Omega_fixed: Whether the Omega weights should be fixed throughout the optimization
        :param theta_fixed: Whether covariance parameters should be fixed throughout the optimization
        :param learn_Omega: How to treat Omega - fixed (from the prior), optimized, or learned variationally
        """
        self.likelihood = likelihood_fun
        self.kernel_type = kernel_type
        self.is_ard = is_ard
        self.feed_forward = feed_forward
        self.q_Omega_fixed = q_Omega_fixed
        self.theta_fixed = theta_fixed
        self.q_Omega_fixed_flag = q_Omega_fixed > 0
        self.theta_fixed_flag = theta_fixed > 0
        self.learn_Omega = learn_Omega
        self.arccosine_degree = kernel_arccosine_degree
        
        self.Dfeatures = 8 * 8 * 64;
        
        self.fastfood_module = tf.load_op_library('fast_food3/fast_food3.so')
        
        ## These are all scalars
        self.num_examples = num_examples
        self.nl = n_layers ## Number of hidden layers
        self.n_Omega = n_layers  ## Number of weigh matrices is "Number of hidden layers"
        self.n_W = n_layers

        ## These are arrays to allow flexibility in the future
        self.n_rff = n_rff * np.ones(n_layers, dtype = np.int64)
        self.df = df * np.ones(n_layers, dtype=np.int64)

        ## Dimensionality of Omega matrices
        if self.feed_forward:
            #self.d_in = np.concatenate([[d_in], self.df[:(n_layers - 1)] + d_in])
            self.d_in = np.concatenate([[self.Dfeatures], self.df[:(n_layers - 1)] + self.Dfeatures])
        else:
            #self.d_in = np.concatenate([[d_in], self.df[:(n_layers - 1)]])
            self.d_in = np.concatenate([[self.Dfeatures], self.df[:(n_layers - 1)]]) # assume that self.d_in include 2^k element
        self.d_out = self.n_rff

        ## Dimensionality of W matrices
        if self.kernel_type == "RBF":
            self.dhat_in = self.n_rff * 2
            self.dhat_out = np.concatenate([self.df[:-1], [d_out]])

        if self.kernel_type == "arccosine":
            self.dhat_in = self.n_rff
            self.dhat_out = np.concatenate([self.df[:-1], [d_out]])
            
        ## Create length-scale and sigma variable of RBF kernel
        ## At the first version, all dimension will have the same ls <==> is_ard = false
        self.llscale0 = tf.constant(0.5 * np.log(self.d_in), tf.float32)
        self.log_theta_lengthscale = tf.Variable(self.llscale0, name="log_theta_lengthscale")
        self.log_theta_sigma2 = tf.Variable(tf.zeros([n_layers]), name="log_theta_sigma2")
        
        ## Create random_vector
        self.B, self.P, self.G, self.s = self.init_random_vector()
        
        ## Set the prior over weights
        self.prior_mean_W, self.log_prior_var_W = self.get_prior_W()

        self.mean_W, self.log_var_W = self.init_posterior_W()
        
        ## Initialize filters and bias of CNN structures
        self.filters1, self.bias1, self.filters2, self.bias2 = self.init_param_CNN()

        ## Set the number of Monte Carlo samples as a placeholder so that it can be different for training and test
        self.mc =  tf.placeholder(tf.int32) 

        ## Batch data placeholders
        Din = d_raw_in
        Dout = d_out
        self.X = tf.placeholder(tf.float32, [None, Din])
        self.Y = tf.placeholder(tf.float32, [None, Dout])

        ## Builds whole computational graph with relevant quantities as part of the class
        self.loss, self.kl, self.ell, self.layer_out = self.get_nelbo()

        ## Initialize the session
        self.session = tf.Session()

    ## Definition of a prior over W - these are standard normals
    def get_prior_W(self):
        prior_mean_W = tf.zeros(self.n_W)
        log_prior_var_W = tf.zeros(self.n_W)
        return prior_mean_W, log_prior_var_W

    ## Function to initialize the posterior over W
    def init_posterior_W(self):
        mean_W = [tf.Variable(tf.zeros([self.dhat_in[i], self.dhat_out[i]]), name="q_W") for i in range(self.n_W)]
        log_var_W = [tf.Variable(tf.zeros([self.dhat_in[i], self.dhat_out[i]]), name="q_W") for i in range(self.n_W)]

        return mean_W, log_var_W
    
    ## Init random vector
    def init_random_vector(self):
        B = []
        P = []
        G = []
        s = []
        for i in range(self.nl):
            B.append(tf.Variable(random_vector.create_binary_scaling_vector(self.d_in[i]), trainable=False))
            P.append(tf.Variable(random_vector.create_permutation(self.d_in[i]), trainable=False))
            G.append(tf.Variable(tf.random_normal([self.d_in[i], 1]), trainable=False))
            #S.append(tf.Variable(tf.random_normal([self.d_in[i], 1]), trainable=False))
            s.append(tf.Variable(random_vector.create_chi_vector(self.d_in[i]), trainable=False))
        return B, P, G, s
        
    ## Function to initialize the filters and bias of CNN structure
    def init_param_CNN(self):
        filters1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), tf.float32, name="filters1")
        bias1 = tf.Variable(tf.truncated_normal([32], stddev=0.1), tf.float32, name="bias1")
        filters2= tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), tf.float32, name="filters2")
        bias2 = tf.Variable(tf.truncated_normal([64], stddev=0.1), tf.float32, name="bias2")
        return filters1, bias1, filters2, bias2

    ## Function to compute the KL divergence between priors and approximate posteriors over model parameters (W only) 
    def get_kl(self):
        kl = 0
        for i in range(self.n_W):
            kl = kl + utils.DKL_gaussian(self.mean_W[i], self.log_var_W[i], self.prior_mean_W[i], self.log_prior_var_W[i])
        return kl

    ## Returns samples from approximate posterior over W
    def sample_from_W(self):
        W_from_q = []
        for i in range(self.n_W):
            z = utils.get_normal_samples(self.mc, self.dhat_in[i], self.dhat_out[i])
            W_from_q.append(tf.add(tf.mul(z, tf.exp(self.log_var_W[i] / 2)), self.mean_W[i]))
        return W_from_q
    
    ## padding zero to x_images from 28 * 28 to 32 * 32
    ## Input:
    ##  + x_images: [batch_size, 28, 28, 1]
    ## Output:
    ##  + x_images_zero_padding: [batch_size, 32, 32, 1]
    def zero_padding_32_32(self, x_images):
        x_images_shape = tf.shape(x_images)
        batch_size = tf.slice(x_images_shape, [0], [1])
        batch_size = tf.reshape(batch_size, [])
        zero_padding_row = tf.zeros([batch_size, 2, 28, 1])
        zero_padding_column = tf.zeros([batch_size, 32, 2, 1])
        x_images_zero_padding = tf.concat(1, [zero_padding_row, x_images])
        x_images_zero_padding = tf.concat(1, [x_images_zero_padding, zero_padding_row])
        x_images_zero_padding = tf.concat(2, [zero_padding_column, x_images_zero_padding])
        x_images_zero_padding = tf.concat(2, [x_images_zero_padding, zero_padding_column])
        return x_images_zero_padding
    
    ## Returns the expected log-likelihood term in the variational lower bound 
    def get_ell(self):
        Din = self.d_in[0]
        MC = self.mc
        N_L = self.nl
        X = self.X
        Y = self.Y
        batch_size = tf.shape(X)[0] # This is the actual batch size when X is passed to the graph of computations
        
        ## CNN structure
        x_images = tf.reshape(X, [-1, 28, 28, 1]) #[batch_size, 28, 28, 1]
        x_images_zero_padding = self.zero_padding_32_32(x_images) #[batch_size, 32, 32, 1]
        conv_bias1 = tf.add(tf.nn.conv2d(x_images_zero_padding, self.filters1, strides=[1,1,1,1], padding = "SAME"), self.bias1) #[batch_size, 32, 32, 32]
        relu1 = tf.nn.relu(conv_bias1) #[batch_size, 32, 32, 32]
        subsampling1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") #[batch_size, 16, 16, 32]
        conv_bias2 = tf.add(tf.nn.conv2d(subsampling1, self.filters2, strides=[1,1,1,1], padding="SAME"), self.bias2) #[batch_size, 16, 16, 64]
        relu2 = tf.nn.relu(conv_bias2) #[batch_size, 16, 16, 64]
        subsampling2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") #[batch_size, 8, 8, 64]
        featuresCNN = tf.reshape(subsampling2, [-1, 8 * 8 * 64]) #[batch_size, 8 * 8 * 64]
        
        ## The representation of the information is based on 3-dimensional tensors (one for each layer)
        ## Each slice [i,:,:] of these tensors is one Monte Carlo realization of the value of the hidden units
        ## At layer zero we simply replicate the input matrix X self.mc times
        self.layer = []
        self.layer.append(tf.mul(tf.ones([self.mc, batch_size, self.Dfeatures]), featuresCNN))

        ## Forward propagate information from the input to the output through hidden layers
        W_from_q = self.sample_from_W()
        # TODO: basis features should be in a different class
        for i in range(N_L):
            #layer_times_Omega = tf.batch_matmul(self.layer[i], Omega_from_q[i])  # X * Omega
            
            layer_times_Omega = tf.zeros([1])
            if (i == 0):
                x_tensors = tf.transpose(featuresCNN) #[Dfeatures, batch_size]
                layer_times_Omega = self.fastfood_module.fast_food3(x_tensors, self.B[0], self.P[0], self.G[0], self.s[0]) #[Dfeatures, batch_size]
                layer_times_Omega = tf.transpose(layer_times_Omega) #[batch_size, Dfeatures]
                layer_times_Omega = (1.0 / (tf.exp(self.log_theta_lengthscale[i]) * tf.sqrt(1.0 * self.d_in[i]))) * layer_times_Omega
                layer_times_Omega = tf.expand_dims(layer_times_Omega, 0)
                layer_times_Omega = tf.tile(layer_times_Omega, [self.mc, 1, 1])
            else:
                x_tensors = tf.reshape(self.layer[i], [self.mc * batch_size, -1])
                x_tensors = tf.transpose(x_tensors)
                layer_times_Omega = self.fastfood_module.fast_food3(x_tensors, self.B[i], self.P[i], self.G[i], self.s[i]) #[-1, batch_size * mc]
                layer_times_Omega = tf.transpose(layer_times_Omega) #[batch_size * mc, -1]
                layer_times_Omega = (1.0 / (tf.exp(self.log_theta_lengthscale[i]) * tf.sqrt(1.0 * self.d_in[i]))) * layer_times_Omega
                layer_times_Omega = tf.reshape(layer_times_Omega, [self.mc, batch_size, -1])
            
            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF": 
                # Cluster machine
                Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(2, [tf.cos(layer_times_Omega), tf.sin(layer_times_Omega)])
                # Local machine
                # Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat([tf.cos(layer_times_Omega), tf.sin(layer_times_Omega)], 2)
            if self.kernel_type == "arccosine": 
                if self.arccosine_degree == 0:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(2, [tf.sign(tf.maximum(layer_times_Omega, 0.0))])
                if self.arccosine_degree == 1:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(2, [tf.maximum(layer_times_Omega, 0.0)])
                if self.arccosine_degree == 2:
                    Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / (tf.sqrt(1. * self.n_rff[i])) * tf.concat(2, [tf.square(tf.maximum(layer_times_Omega, 0.0))])

            F = tf.batch_matmul(Phi, W_from_q[i])
            if self.feed_forward and not (i == (N_L-1)): ## In the feed-forward case, no concatenation in the last layer so that F has the same dimensions of Y
                F = tf.concat(2, [F, self.layer[0]])

            self.layer.append(F)

        ## Output layer
        layer_out = self.layer[N_L]

        ## Given the output layer, we compute the conditional likelihood across all samples
        ll = self.likelihood.log_cond_prob(Y, layer_out)

        ## Mini-batch estimation of the expected log-likelihood term
        ell = tf.reduce_sum(tf.reduce_mean(ll, 0)) * self.num_examples / tf.cast(batch_size, "float32")

        return ell, layer_out

    ## Maximize variational lower bound --> minimize Nelbo
    def get_nelbo(self):
        kl = self.get_kl()
        ell, layer_out = self.get_ell()
        nelbo  = kl - ell
        return nelbo, kl, ell, layer_out

    ## Return predictions on some data
    def predict(self, testX, testY, mc_test):
        out = self.likelihood.predict(self.layer_out)
        
        nll = - tf.reduce_sum(-np.log(mc_test) + utils.logsumexp(self.likelihood.log_cond_prob(self.Y, self.layer_out), 0))
        uncertainties = tf.nn.softmax(self.layer_out, dim=-1)
        uncertainties = tf.reduce_mean(uncertainties, 0)
        uncertainties, pred, neg_ll = self.session.run([uncertainties, out, nll], feed_dict={self.X:testX, self.Y: testY, self.mc:mc_test})
        mean_pred = np.mean(pred, 0)
        return mean_pred, neg_ll, uncertainties

    ## Return the list of TF variables that should be "free" to be optimized
    def get_vars_fixing_some(self, all_variables):
        if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == True):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega") and not v.name.startswith("log_theta"))]

        if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == False):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega"))]

        if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == True):
            variational_parameters = [v for v in all_variables if (not v.name.startswith("log_theta"))]

        if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == False):
            variational_parameters = all_variables

        return variational_parameters

    ## Function that learns the deep GP model with random Fourier feature approximation
    def learn(self, trainX, trainY, learning_rate, mc_train, batch_size, n_iterations, optimizer = None, display_step=100, testX=None, testY=None, mc_test=None, loss_function=None, duration = 1000000, less_prints=False):
        total_train_time = 0

        if optimizer is None:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        ## Set all_variables to contain the complete set of TF variables to optimize
        all_variables = tf.trainable_variables()

        ## Define the optimizer
        train_step = optimizer.minimize(self.loss, var_list=all_variables)

        ## Initialize all variables
        init = tf.global_variables_initializer()
        ##init = tf.initialize_all_variables()

        ## Fix any variables that are supposed to be fixed
        train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

        ## Initialize TF session
        self.session.run(init)

        ## Set the folder where the logs are going to be written 
        # summary_writer = tf.train.SummaryWriter('logs/', self.session.graph)
        summary_writer = tf.summary.FileWriter('logs/', self.session.graph)

        if not(less_prints):
            nelbo, kl, ell, _ =  self.session.run(self.get_nelbo(), feed_dict={self.X: data.X, self.Y: data.Y, self.mc: mc_train})
            print("Initial kl=" + repr(kl) + "  nell=" + repr(-ell) + "  nelbo=" + repr(nelbo), end=" ")
            print("  log-sigma2 =", self.session.run(self.log_theta_sigma2))
        
        train_size = len(trainX)
        test_size = len(testX)
        t = 10
        snapshot = t * display_step
        prefix = "mnist_" +  str(train_size) + "train_" + str(test_size) + "test_" + str(self.nl) + "nl_" + str(self.n_rff[0]) + "rff_" + str(self.df[0]) + "df_" \
                  + str(mc_train) + "mctrain_" + str(mc_test) + "mctest_" + str(batch_size) + "bs_" + str(self.theta_fixed) + "tf_" + str(self.feed_forward) + "ff_" \
                  + str(self.learn_Omega) + "Omega" + "_snap_"
        filename = prefix + str(0)
        file = open(filename, 'w')
        cur_iter = open(str(train_size) + "_" + str(0), 'w')
        start_id = 0
        end_id = start_id + batch_size
        ## Present data to DGP n_iterations times
        ## TODO: modify the code so that the user passes the number of epochs (number of times the whole training set is presented to the DGP)
        for iteration in range(n_iterations):
            
            if (iteration > 0) and (iteration % 10 == 0):
                os.rename(str(train_size) + "_" + str(iteration - 10), str(train_size) + "_" + str(iteration))
                
            ## Stop after a given budget of minutes is reached
            if (total_train_time > 1000 * 60 * duration):
                break

            ## Present one batch of data to the DGP
            start_train_time = current_milli_time()
            batch_X = trainX[start_id:end_id]
            batch_Y = trainY[start_id:end_id]
            start_id = start_id + batch_size
            end_id = end_id + batch_size
            if (end_id > train_size):
                start_id = 0
                end_id = batch_size

            monte_carlo_sample_train = mc_train
            if (current_milli_time() - start_train_time) < (1000 * 60 * duration / 2.0): 
                monte_carlo_sample_train = 1

            self.session.run(train_step, feed_dict={self.X: batch_X, self.Y: batch_Y, self.mc: monte_carlo_sample_train})
            total_train_time += current_milli_time() - start_train_time

            ## After reaching enough iterations with Omega fixed, unfix it
            if self.q_Omega_fixed_flag == True:
                if iteration >= self.q_Omega_fixed:
                    self.q_Omega_fixed_flag = False
                    train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

            if self.theta_fixed_flag == True:
                if iteration >= self.theta_fixed:
                    self.theta_fixed_flag = False
                    train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

            ## Display logs every "FLAGS.display_step" iterations
            if iteration % display_step == 0:
                start_predict_time = current_milli_time()

                if less_prints:
                    print("i=" + repr(iteration), end = " ")

                else:
                    nelbo, kl, ell, _ = self.session.run(self.get_nelbo(), feed_dict={self.X: trainX, self.Y: trainY, self.mc: mc_train})
                    print("i=" + repr(iteration)  + "  kl=" + repr(kl) + "  nell=" + repr(-ell)  + "  nelbo=" + repr(nelbo), end=" ")

                    print(" log-sigma2=", self.session.run(self.log_theta_sigma2), end=" ")
                    # print(" log-lengthscale=", self.session.run(self.log_theta_lengthscale), end=" ")
                    # print(" Omega=", self.session.run(self.mean_Omega[0][0,:]), end=" ")
                    # print(" W=", self.session.run(self.mean_W[0][0,:]), end=" ")

                if loss_function is not None:
                    pred, nll_test, uncertainties = self.predict(testX, testY, mc_test)
                    elapsed_time = total_train_time + (current_milli_time() - start_predict_time)
                    err_rate = loss_function.eval(testY, pred)
                    true_false = np.reshape((np.argmax(pred, 1) == np.argmax(testY, 1)), [len(pred), 1])
                    uncertainties = np.concatenate((true_false, uncertainties), axis=-1)
                    print(loss_function.get_name() + "=" + "%.9f" % err_rate, end = " ")
                    print(" nll_test=" + "%.9f" % (nll_test / len(testY)), end = " ")
                    file.write("%d\t%s\t%s\n" % (iteration, err_rate, nll_test / len(testY)))
                    if (iteration >= 0) and (iteration % snapshot == 0):
                        file.close()
                        filename = prefix + str(iteration)
                        file = open(filename, 'w')
                        # Save the uncertainties
                        np.savetxt(prefix + str(iteration) + "_uncer", uncertainties, fmt='%0.5f', delimiter='\t', newline='\n')
                        
                print(" time=" + repr(elapsed_time), end = " ")
                print("")