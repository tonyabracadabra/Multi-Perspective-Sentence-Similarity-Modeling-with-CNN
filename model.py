__author__ = "Xupeng Tong"
__copyright__ = "Copyright 2017, Text similarity Mesurement with Multi-Perspective CNN"
__email__ = "xtong@andrew.cmu.edu"

import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from keras.layers import Input, Lambda, Dense
from keras.layers.convolutional import Convolution2D
from tensorflow.contrib.distributions import kl, Multinomial
from tensorflow.contrib.layers import convolution2d, fully_connected

# Import all the layers from utils file
from layers import *
from utils import *

tf.app.flags.DEFINE_integer('d', 100, 'The dimension of the word embedding')
tf.app.flags.DEFINE_string('max_rating', 5, 'The max rating of the dataset')
tf.app.flags.DEFINE_integer('num_filters_A', 200, 'The number of filters in block A')
tf.app.flags.DEFINE_integer('num_filters_B', 20, 'The number of filters in block B')
tf.app.flags.DEFINE_string('gpu_fraction', '1/2', 'define the gpu fraction used')
tf.app.flags.DEFINE_integer('n_hidden', 150, 'number of hidden units in the fully connected layer')
tf.app.flags.DEFINE_string('optim_type', 'adam', 'optimizer')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of epochs to be trained')
tf.app.flags.DEFINE_integer('data_path', '.', 'Path for the data')
tf.app.flags.DEFINE_integer('batch_size', 30, 'Path for the data')


# adam optimizer
tf.app.flags.DEFINE_float('lr', 1e-2, 'learning rate')
tf.app.flags.DEFINE_float('lambda', 1e-4, 'regularization parameter')

conf = tf.app.flags.FLAGS
wss = [1, 2, 3, sys.maxint]

"""
Configure all placeholders to be fed
"""
# Defines the placeholder for the sentence pair
# Sentences are represented as word embeddings, one of the Glove, Paragram-Phrase, Word2Vec etc
sent_0 = tf.placeholder(shape=(None, conf.d), dtype='float64', name='sent_input_0')
sent_1 = tf.placeholder(shape=(None, conf.d), dtype='float64', name='sent_input_1')
# True Distribution vector, together with p, KL divergence can be calculated and used as loss to be minimized
p_ = tf.placeholder(shape=(1, conf.max_rating), dtype='float64', name='target_distribution')

sentences = [sent_0, sent_1]

"""
Step 1

Attention-Based Input Interaction Layer
"""
atten_embeds = AttentionInputLayer(sentences).atten_embeds

"""
Step 2 + 3

Multi-Perspective Sentence Model + Structured Similarity Measurement Layer

(Two algorithms are implemented and either one could be used for sentence modelling)
"""

setence_model = SentenceModelingLayer(conf, atten_embeds, wss)

fea_h = setence_model.horizontal_comparison()

# or alternatively,

fea_v = setence_model.vertical_comparison()

"""
Step 4

Output: Similarity Score
"""

ss = SimilarityScoreLayer(fea_h, conf)
p = ss.generate_p()

# Difine the loss as kl divergence, more losses will be supported
kl_loss = kl_divergence(p_, p)

# Define regularization over all convolutional/fully connected layers
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

overal_loss = kl_loss + reg_losses

train_step = tf.train.AdamOptimizer(conf.lr).minimize(overal_loss)

# Initializing the variables
init = tf.global_variables_initializer()

sentence_pairs = np.load(conf.data_path)

with tf.Session as sess:
	sess.run(init)

	for epoch in range(conf.epochs):
        avg_cost = 0.
        # Loop over all batches
        for sentence_pair in sentence_pairs:
            sent_0_val, sent_1_val, y = sentence_pair
            # Convert the target into distribution
            p_val = sparse_target_distribution(y)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_step, overal_loss], feed_dict={sent_0: sent_0_val, 
            									sent_1: sent_1_val, p_: p_val})
            # Compute average loss
            avg_cost += c / len(sentence_pairs)
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

	sess.run(train_step)