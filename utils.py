__author__ = "Xupeng Tong"
__copyright__ = "Copyright 2016, Text similarity"
__email__ = "xtong@andrew.cmu.edu"


"""

"" It is applied over raw word embeddings of input sentences to generate re-weighted word embeddings. 
The attention-based re-weightings can guide the focus of the MPCNN model onto important input words. ""

Proposed by the paper

He, Hua, et al. 
"UMD-TTIC-UW at SemEval-2016 Task 1: Attention-Based Multi-Perspective Convolutional Neural Networks for 
Textual Similarity Measurement."

"""
def input_layer(sentences):
    # Dimensions of two sentence embeddings are supposed to be same
    ds = [sentence.get_shape()[1].value for sentence in sentences]
    assert ds[0] == ds[1]
    
    # norm2 by row
    norm2 = lambda X : K.expand_dims(K.sqrt(K.sum(X ** 2, 1)))
    # Cosine distance by its best definition
    cosine = lambda X, Y : K.dot(X, K.transpose(Y))/norm2(X)/K.transpose(norm2(Y))

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

    return atten_embeds



wss = [1,2,3]

# Let the number of filters be 
num_filters = 2*2*d

"""
Multi-Perspective Sentence Model

Two algorithms are used here
"""

# Algorithm 1 Horizontal Comparison

# In the horizontal direction, each equal-sized max/min/mean group is extracted as a vector and 
# is compared to the corresponding one for the other sentence.

def sentence_algo1(atten_embeds):
    fea_h = []
    for i, pooling in enumerate([K.max, K.min, K.mean]):
        regM0, regM1 = [], []
        for j, ws in enumerate(wss):
            for k, atten_embed in enumerate(atten_embeds):
                # Building block A, moving the window across the whole length of the word embedding
                conv = convolution2d(atten_embed, num_filters, kernel_size=[ws, 2*d], stride=[1,1], padding='VALID')
                conv = tf.squeeze(conv, axis=[0,2])
                if k == 0:
                    regM0.append(pooling(conv, 0))
                else:
                    regM1.append(pooling(conv, 0))
        regM0, regM1 = tf.pack(regM0), tf.pack(regM1)

        for n in xrange(num_filters):
            fea_h.append(comU2(regM0[:,n], regM1[:,n]))

    fea_h = K.expand_dims(K.flatten(fea_h),0)
    
    return fea_h


# Algorithm 2 Vertical Comparison

def sentence_algo2(atten_embeds):
    fea_a = []
    for i, pooling in enumerate([K.max, K.min, K.mean]):
        regM0, regM1 = [], []
        atten_embed_0, atten_embed_1 = atten_embeds
        # Building block A, moving the window across the whole length of the word embedding
        for j_0, ws_0 in enumerate(wss):
            oG0A = convolution2d(atten_embed_0, num_filters, kernel_size=[ws_0, 2*d], stride=[1,1], padding='VALID')
            for j_1, ws_1 in enumerate(wss):
                oG1A = convolution2d(atten_embed_1, num_filters, kernel_size=[ws_1, 2*d], stride=[1,1], padding='VALID')
                fea_a.append(comU1(oG1A, oG2A))

        for b, ws in enumerate(wss):
            oG0B = convolution2d(atten_embed_0, num_filters, kernel_size=[1, ws], stride=[1,1], padding='VALID')
            oG1B = convolution2d(atten_embed_1, num_filters, kernel_size=[1, ws], stride=[1,1], padding='VALID')

            for n in xrange(num_filters_B):
                fea_b.append(comU1(oG0B[:,n], oG0B[:,n]))

        regM0, regM1 = tf.pack(regM0), tf.pack(regM1)


    fea_B = K.expand_dims(K.flatten(fea_B),0)
    
    return fea_A


"""
Similarity Comparison Units

"Cosine distance (cos) measures the distance of
two vectors according to the angle between them,
while L2 Euclidean distance (L2Euclid) and
element-wise absolute difference measure magnitude
differences."

"""

def comU1(vec0, vec1):
    cos_dist = K.sum(vec0 * vec1)/K.sqrt(K.sum(vec0 ** 2))/K.sqrt(K.sum(vec1 ** 2))
    l2_dist = K.sqrt(K.sum(K.square(vec0 - vec1)))
    l1_dist = K.abs(vec0 - vec1)

    result = tf.pack([cos_dist, l2_dist, l1_dist])
    
    return result

def comU2(vec0, vec1):
    cos_dist = K.sum(vec0 * vec1)/K.sqrt(K.sum(vec0 ** 2))/K.sqrt(K.sum(vec1 ** 2))
    l2_dist = K.sqrt(K.sum(K.square(vec0 - vec1)))

    result = tf.pack([cos_dist, l2_dist])
    
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

