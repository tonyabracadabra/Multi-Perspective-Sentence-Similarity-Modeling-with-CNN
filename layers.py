__author__ = "Xupeng Tong"
__copyright__ = "Copyright 2017, Text similarity Mesurement with Multi-Perspective CNN"
__email__ = "xtong@andrew.cmu.edu"


from utils import *
import tensorflow as tf
from keras import backend as K
from tensorflow.contrib.layers import convolution2d, fully_connected, l2_regularizer

"""

"" It is applied over raw word embeddings of input sentences to generate re-weighted word embeddings. 
The attention-based re-weightings can guide the focus of the MPCNN model onto important input words. ""

Proposed by the paper

He, Hua, et al. 
"UMD-TTIC-UW at SemEval-2016 Task 1: Attention-Based Multi-Perspective Convolutional Neural Networks for 
Textual Similarity Measurement."

"""

class AttentionInputLayer:
    def __init__(self, sentences):
        # Dimensions of two sentence embeddings are supposed to be same
        ds = [sentence.get_shape()[1].value for sentence in sentences]
        assert ds[0] == ds[1]
        
        # norm2 by row
        norm2 = lambda X : K.expand_dims(K.sqrt(K.sum(X ** 2, 1)))
        # Cosine distance by its best definition
        cosine = lambda X, Y : K.dot(X, K.transpose(Y))/norm2(X)/K.transpose(norm2(Y))

        # Cosine similarity matrix
        D = cosine(sentences[0], sentences[1])
        
        # Attention weight vectors for sentence 0 and 1
        As = [K.softmax(K.expand_dims(K.sum(D,i),1)) for i in [1,0]]
        
        # Concatenating the original sentence embedding to the cosine similarity, 
        # force the dimension to be expanded from d to 2*d
        atten_embeds = []
        for i in xrange(2):
            atten_embed = tf.concat(1, [sentences[i], As[i] * sentences[i]])
            atten_embed = K.expand_dims(atten_embed, 0)
            atten_embed = K.expand_dims(atten_embed, 3)
            atten_embeds.append(atten_embed)

        self.atten_embeds = atten_embeds

"""
Multi-Perspective Sentence Model

Two algorithms are used here
"""

# Algorithm 1 Horizontal Comparison

# In the horizontal direction, each equal-sized max/min/mean group is extracted as a vector and 
# is compared to the corresponding one for the other sentence.

class SentenceModelingLayer:

    def __init__(self, conf, atten_embeds, wss):
        self.conf = conf
        self.atten_embeds = atten_embeds
        self.regularizer = l2_regularizer(conf.lambda)
        self.fea_h, fea_v = None, None
        self.wss = wss

    # Algorithm 1 Horizontal Comparison
    def horizontal_comparison(self):
        if self.fea_h is not None:
            return self.fea_h

        fea_h = []
        with tf.variable_scope("algo_1"):
            for i, pooling in enumerate([K.max, K.min, K.mean]):
                regM0, regM1 = [], []
                for j, ws in enumerate(self.wss):
                    for k, atten_embed in enumerate(atten_embeds):
                        # Working with building block A, moving the window across the whole length of the word embedding
                        conv = self.__building_block_A(atten_embed, ws_0)
                        conv = tf.squeeze(conv, axis=[0,2])
                        if k == 0:
                            regM0.append(pooling(conv, 0))
                        else:
                            regM1.append(pooling(conv, 0))
                regM0, regM1 = tf.pack(regM0), tf.pack(regM1)

                for n in xrange(num_filters):
                    fea_h.append(comU2(regM0[:,n], regM1[:,n]))

            fea_h = K.expand_dims(K.flatten(fea_h),0)

            self.fea_h = fea_h
        
        return fea_h

    # Algorithm 2 Vertical Comparison
    def vertical_comparison(self):
        if self.fea_v is not None:
            return self.fea_v

        fea_a, fea_b = [], []

        with tf.variable_scope("algo_2"):
            for i, pooling in enumerate([K.max, K.min, K.mean]):
                atten_embed_0, atten_embed_1 = self.atten_embeds

                # Working with building block A, moving the window across the whole length of the word embedding
                for j_0, ws_0 in enumerate(self.wss):
                    oG0A = self.__building_block_A(atten_embed_0, ws_0, d)
                    for j_1, ws_1 in enumerate(wss):
                        oG1A = self.__building_block_A(atten_embed_1, ws_1, d)
                        fea_a.append(comU1(oG0A, oG1A))

                # Working with building block B, the per dimensional CNN
                for b, ws in enumerate(self.wss[:-1]):
                    oG0B = self.__building_block_B(atten_embed_0, ws)
                    oG0B = tf.pack([pooling(conv,0) for conv in oG0B])

                    oG1B = self.__building_block_B(atten_embed_1, ws)
                    oG1B = tf.pack([pooling(conv,0) for conv in oG1B])
                    
                    for n in xrange(num_filters_B):
                        fea_b.append(comU1(oG0B[:,n], oG1B[:,n]))
            
            # Concatenate them up! Oops that's a lot...
            fea_v = K.expand_dims(tf.concat(0, map(K.flatten, [fea_a, fea_b])),0)

            self.fea_a, self.fea_b, self.fea_v = fea_a, fea_b, fea_v
        
        return fea_v

    
    """
    Function that given a input (4 dimensional tensor), returns the hollistic CNN of building block A

    """
    def __building_block_A(self, input, ws):
        with tf.variable_scope("building_block_A"):
            conv = convolution2d(input, self.conf.num_filters_A, kernel_size=[ws, 2*self.conf.d], stride=[1,1], padding='VALID', \
                                    weights_regularizer=self.regularizer, biases_regularizer=self.regularizer)
        return conv

    """
    Function that given a input (4 dimensional tensor), returns the row-wise components of building block B
    Note that the CNN at each dimension does not share parameters, thus after pooling, the return size == dimension,
    where we can start from comparing the generated vectors in the depth of num_filter_B

    """
    def __building_block_B(self, input, ws):
        # Dimension where we want to iteration through with multiple 1D CNN
        dimension = input.get_shape()[2].value

        # Stores the 1d conv output
        convs = []
        # Per dimension iteration
        with tf.variable_scope("building_block_B"):
            for d in xrange(dimension):
                conv = convolution2d(tf.expand_dims(input[:,:,d,:],1), self.conf.num_filters_B, kernel_size=[1,ws], stride=[1,1], \
                    padding='VALID', weights_regularizer=self.regularizer, biases_regularizer=self.regularizer)
                # Removing the dimension with 1
                conv = tf.squeeze(conv, axis=[0,1])
                convs.append(conv)

        return convs

"""
Given the feature extracted by the sentence modeling by either of the two algorithms,
pass it to the fully connected layer to generate the predicted distribution p

"""

class SimilarityScoreLayer:

    def __init__(self, input, conf):
        self.conf = conf
        self.output = None

    def generate_p(self):
        if self.output is not None:
            return self.output

        linear_layer = fully_connected(input, self.conf.n_hidden, activation_fn=K.tanh, \
                        weights_regularizer=l2_regularizer, biases_regularizer=l2_regularizer)
        output = fully_connected(linear_layer, 5, activation_fn=tf.nn.log_softmax, \
                        weights_regularizer=l2_regularizer, biases_regularizer=l2_regularizer)

        self.output = output

    return output
