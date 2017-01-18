__author__ = "Xupeng Tong"
__copyright__ = "Copyright 2017, Text similarity Mesurement with Multi-Perspective CNN"
__email__ = "xtong@andrew.cmu.edu"

import tensorflow as tf
from keras import backend as K

"""
Similarity Comparison Units

"Cosine distance (cos) measures the distance of
two vectors according to the angle between them,
while L2 Euclidean distance (L2Euclid) and
element-wise absolute difference measure magnitude
differences."

cosine_distance(x, y) = (x^T * y) / (norm2(x) * norm2(y))
l2_distance(x, y) = \sqrt \sum (x - y)^2
l1_distance(x, y) = \sum \abs (x - y)

"""

cos_dist = lambda x, y : K.sum(x * y)/K.sqrt(K.sum(x ** 2))/K.sqrt(K.sum(y ** 2))
l2_dist = lambda x, y : K.sqrt(K.sum(K.square(x - y)))
l1_dist = lambda x, y : K.sum(K.abs(x - y))

def comU1(vec_0, vec_1):
    result = tf.pack([cos_dist(vec_0, vec_1), l2_dist(vec_0, vec_1), l1_dist(vec_0, vec_1)])

    return result

def comU2(vec0, vec1):
    result = tf.pack([cos_dist(vec_0, vec_1), l2_dist(vec_0, vec_1)])
    
    return result

"""
Used for generating the sparse targe distribution originally by the paper

Tai, Kai Sheng, Richard Socher, and Christopher D. Manning. 
"Improved semantic representations from tree-structured long short-term memory networks." arXiv preprint arXiv:1503.00075 (2015).

"""

def sparse_target_distribution(y):
    y_floor = int(np.floor(y))
    p = np.zeros(5)
    for i in xrange(y):
        if i == y_floor:
            p[i] = y - y_floor
        elif i == y_floor - 1:
            p[i] = y - y_floor + 1
    
    return p

"""
Given two distributions p and q, calculate the kl-divergence

"""
def kl_divergence(p, q):
    cross_entropy = -tf.reduce_sum(p * tf.log(q))
    entropy = -tf.reduce_sum(p * tf.log(p + 0.00001))
    kl_div = cross_entropy - entropy

    return kl_div

