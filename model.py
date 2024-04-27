import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import math


class AttentionMap(tf.keras.layers.Layer):
    def __init__(self, kernel_size
    , **kwargs):
        super(AttentionMap, self).__init__(**kwargs)
        self.kernel_size = kernel_size

        self.K = tf.keras.layers.Dense(kernel_size)
        self.Q = tf.keras.layers.Dense(kernel_size)
        self.V = tf.keras.layers.Dense(kernel_size)

    def call(self,conv_local,conv_global):
        K = self.K(conv_local)
        V = self.V(conv_local) # not sure if we need this 
        Q = self.Q(conv_global)

        weighted_sum = tf.matmul(Q,K,transpose_b=True)  
        d_k = K.shape[2]
        weighted_sum= weighted_sum/tf.sqrt(float(d_k))

        attention_matrix = tf.nn.softmax(weighted_sum)
        attention_map_output = tf.matmul(attention_matrix,V)
        return attention_map_output


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self,kernel_size, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.upfactor = 1
        self.w_local = tf.keras.layers.Conv2D(filters=8, kernel_size=kernel_size, padding='same')
        self.w_global = tf.keras.layers.Conv2D(filters=8, kernel_size=kernel_size, padding='same')
        self.attention_module = AttentionMap(kernel_size=kernel_size)
        self.final_conv = tf.keras.layers.Conv2D(filters=8, kernel_size=kernel_size, padding='same')
        self.normalize = tf.keras.layers.Softmax()
 
    def call(self, x_local, x_global):
        conv_local = self.w_local(x_local)
        conv_global = self.w_global(x_global)

        #Interpolation -- needs work
        if self.upfactor > 1:
              conv_global = tf.image.resize(conv_global, size=(conv_local.shape[1], conv_local.shape[2]))
            #   upsampling_layer = tf.keras.layers.UpSampling2D(size=(25/7, 25/7), data_format='channels_last')
            #   conv_global = upsampling_layer(conv_global)
        #conv_global_new = tf.zeros_like(conv_local)
        #for i in range (0, conv_global.shape[0]
        #for i in range(0, conv_local.shape[0], 2):
        #for j in range(0, conv_local.shape[1], 2):
        #conv_global_new[i, j] = conv_global[int(i/2), int(j/2)]

         # # Alignment scores. Pass them through tanh function
        # e = K.tanh(K.dot(x,self.W)+self.b)
        # # Remove dimension of size 1
        # e = K.squeeze(e, axis=-1)   
        # # Compute the weights
        # alpha = K.softmax(e)
        # # Reshape to tensorFlow format
        # alpha = K.expand_dims(alpha, axis=-1)
        # # Compute the context vector
        # context = x * alpha
        # context = K.sum(context, axis=1)
        # return context

        attention_map_output = self.attention_module(conv_local,conv_global)
        attention_layer_output = self.final_conv(attention_map_output)
        return attention_layer_output


class Model(tf.keras.Model):
    def __init__(self, input_size, num_epoch, kernel_size):
        super(Model, self).__init__()
        #     self.filters_1 = tf.Variable(tf.random.truncated_normal([2, 2, input_size[3], kernel_size
        # ],dtype=tf.float32,
        # stddev=1e-1))
        self.num_epoch = num_epoch
        self.conv1 = tf.keras.layers.Conv2D(filters=16,kernel_size=kernel_size, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=16,kernel_size=kernel_size, padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=16,kernel_size=kernel_size, padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=16,kernel_size=kernel_size, padding="same")
        self.attentionlayer = AttentionBlock(kernel_size=kernel_size)
        self.dense1_classification  = tf.keras.layers.Dense(kernel_size)
        self.leaky_relu = tf.keras.layers.LeakyReLU(0.3)
        self.dense2_classification  = tf.keras.layers.Dense(kernel_size)
        self.dense3_classification  = tf.keras.layers.Dense(4)

    def call(self, inputs):
        c1 = self.conv1(inputs)
        r1 = self.leaky_relu(c1)
        p1 = tf.nn.max_pool(r1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        c2 = self.conv2(p1)
        r2 = self.leaky_relu(c2)
        p2 = tf.nn.max_pool(r2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        c3 = self.conv3(p2)
        r3 = self.leaky_relu(c3)
        p3 = tf.nn.max_pool(r3,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        c4 = self.conv4(p3)
        r4 = self.leaky_relu(c4)
        p4 = tf.nn.max_pool(r4,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        self.attentionlayer.upfactor = p2.shape[1] // p4.shape[1]

        attention_output = self.attentionlayer(p2, p4)
        attention_output = tf.reshape(attention_output, shape=(attention_output.shape[0], -1))
        d1 = self.dense1_classification(attention_output)
        d2 = self.dense2_classification(d1)
        d3 = self.dense3_classification(d2)
        return d3
    
    def loss(self, logits, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels,logits)
        return tf.reduce_mean(loss)
    
    def accuracy(self, logits, labels):
        
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))