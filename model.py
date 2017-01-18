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
from utils import *

# Dimension of the original word embedding
d = 10
# The max rating of the dataset, defines the length of the sparse target distribution vector
max_rating = 5

# Defines the placeholder for the sentence pair
# Sentences are represented as word embeddings, one of the Glove, Paragram-Phrase, Word2Vec etc
sent_0 = tf.placeholder(shape=(None,d), dtype='float64', name='sent_input_0')
sent_1 = tf.placeholder(shape=(None,d), dtype='float64', name='sent_input_1')

# True Distribution vector, together with p, KL divergence can be calculated and used as loss to be minimized
p_ = tf.placeholder(shape=(1, max_rating), dtype='float64', name='target_distribution')

sentences = [sent_0, sent_1]

"""
Step 1

Attention-Based Input Interaction Layer
"""
atten_embeds = input_layer(sentences)

"""
Step 2 + 3

Multi-Perspective Sentence Model + Structured Similarity Measurement Layer

(Two algorithms are implemented and either one could be used for sentence modelling)
"""

fea_A = sentence_algo1(atten_embeds)

"""
Step 4

Output: Similarity Score
"""

n_hidden = 150
p = fully_connected(fully_connected(fea_A, n_hidden, activation_fn=tf.nn.tanh), 5, activation_fn=tf.nn.log_softmax)
# p_ = sparse_target_distribution(y)

cross_entropy = -tf.reduce_sum(p_ * tf.log(p))
entropy = -tf.reduce_sum(p_ * tf.log(p_ + 0.00001))
kl_divergence = cross_entropy - entropy

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(kl_divergence)

with tf.Session as sess:
	sess.run(train_step)





