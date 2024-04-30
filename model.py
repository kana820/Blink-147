import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import math

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', use_bias=False)
    def call(self, l, g):
        N, W, H, C = tf.shape(l)
        c = self.op(tf.concat(l, g))  # batch_sizexWxHx1
        c = tf.squeeze(c, axis=-1)
        # N, C, W, H = l.size()
        # c = self.op(l+g) # batch_sizex1xWxH

        if self.normalize_attn:
            a = tf.nn.softmax(tf.reshape(c, [N, -1]), axis=1)  # batch_sizex(W*H)
            a = tf.reshape(a, [N, W, H, 1])  # batch_sizexWxHx1
        else:
            a = tf.sigmoid(c)
            a = tf.expand_dims(a, axis=-1)  # batch_sizexWxHx1
        # if self.normalize_attn:
        #     a = tf.nn.softmax(c.reshape(N,1,-1), dim=2).reshape(N,1,W,H)
        # else:
        #     a = tf.nn.sigmoid(c)
        g = tf.multiply(a, l)
        if self.normalize_attn:
            g = tf.reduce_sum(g, axis=[1, 2])  # batch_sizexC
        else:
            g = tf.reduce_mean(g, axis=[1, 2], keepdims=True)  # batch_sizex1x1xC
            g = tf.squeeze(g, axis=[1, 2])  # batch_sizexC
        
        return tf.expand_dims(c, axis=-1), g
        # g = tf.multiply(a.expand_as(l), l)
        # if self.normalize_attn:
        #     g = g.reshape(N,C,-1).sum(dim=2) # batch_sizexC
        # else:
        #     g = tf.keras.layers.AveragePooling2D(g, (1,1)).reshape(N,C)
        # return c.reshape(N,1,W,H), g


class ProjectorBlock(tf.keras.layers.Layer):
    def __init__(self, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = tf.keras.layers.Conv2D(filters=out_features, kernel_size=1, padding='valid', use_bias=False)
        
    def call(self, inputs):
        return self.op(inputs)

# class AttentionMap(tf.keras.layers.Layer):
#     def __init__(self, kernel_size
#     , **kwargs):
#         super(AttentionMap, self).__init__(**kwargs)
#         self.kernel_size = kernel_size

#         self.K = tf.keras.layers.Dense(kernel_size)
#         self.Q = tf.keras.layers.Dense(kernel_size)
#         self.V = tf.keras.layers.Dense(kernel_size)

#     def call(self,conv_local,conv_global):
#         K = self.K(conv_local)
#         V = self.V(conv_local) # not sure if we need this 
#         Q = self.Q(conv_global)

#         weighted_sum = tf.matmul(Q,K,transpose_b=True)  
#         d_k = K.shape[2]
#         weighted_sum= weighted_sum/tf.sqrt(float(d_k))

#         attention_matrix = tf.nn.softmax(weighted_sum)
#         attention_map_output = tf.matmul(attention_matrix,V)
#         return attention_map_output


# class AttentionBlock(tf.keras.layers.Layer):
#     def __init__(self,kernel_size, **kwargs):
#         super(AttentionBlock, self).__init__(**kwargs)
#         self.upfactor = 1
#         self.w_local = tf.keras.layers.Conv2D(filters=8, kernel_size=kernel_size, padding='same')
#         self.w_global = tf.keras.layers.Conv2D(filters=8, kernel_size=kernel_size, padding='same')
#         self.attention_module = AttentionMap(kernel_size=kernel_size)
#         self.final_conv = tf.keras.layers.Conv2D(filters=8, kernel_size=kernel_size, padding='same')
#         self.normalize = tf.keras.layers.Softmax()
 
#     def call(self, x_local, x_global):
#         conv_local = self.w_local(x_local)
#         conv_global = self.w_global(x_global)

#         #Interpolation -- needs work
#         if self.upfactor > 1:
#               conv_global = tf.image.resize(conv_global, size=(conv_local.shape[1], conv_local.shape[2]))
#             #   upsampling_layer = tf.keras.layers.UpSampling2D(size=(25/7, 25/7), data_format='channels_last')
#             #   conv_global = upsampling_layer(conv_global)
#         #conv_global_new = tf.zeros_like(conv_local)
#         #for i in range (0, conv_global.shape[0]
#         #for i in range(0, conv_local.shape[0], 2):
#         #for j in range(0, conv_local.shape[1], 2):
#         #conv_global_new[i, j] = conv_global[int(i/2), int(j/2)]

#          # # Alignment scores. Pass them through tanh function
#         # e = K.tanh(K.dot(x,self.W)+self.b)
#         # # Remove dimension of size 1
#         # e = K.squeeze(e, axis=-1)   
#         # # Compute the weights
#         # alpha = K.softmax(e)
#         # # Reshape to tensorFlow format
#         # alpha = K.expand_dims(alpha, axis=-1)
#         # # Compute the context vector
#         # context = x * alpha
#         # context = K.sum(context, axis=1)
#         # return context

#         attention_map_output = self.attention_module(conv_local,conv_global)
#         attention_layer_output = self.final_conv(attention_map_output)
#         return attention_layer_output


class Model(tf.keras.Model):
    def __init__(self, input_size, num_epoch, kernel_size):
        super(Model, self).__init__()
        #     self.filters_1 = tf.Variable(tf.random.truncated_normal([2, 2, input_size[3], kernel_size
        # ],dtype=tf.float32,
        # stddev=1e-1))
        self.num_epoch = num_epoch
        self.conv1 = tf.keras.layers.Conv2D(filters=16,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv2 = tf.keras.layers.Conv2D(filters=16,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv3 = tf.keras.layers.Conv2D(filters=16,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv4 = tf.keras.layers.Conv2D(filters=16,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME")

        # self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)

        # self.flatten = tf.keras.layers.Flatten()
        # self.attention = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")
        # Conv2D(1, (1, 1), activation='sigmoid')(x)
        # x = multiply([x, attention])
        # self.attentionlayer = AttentionBlock(kernel_size=kernel_size)

        self.dense1_classification  = tf.keras.layers.Dense(kernel_size, activation=tf.keras.layers.LeakyReLU())
        self.dense2_classification  = tf.keras.layers.Dense(kernel_size, activation=tf.keras.layers.LeakyReLU())
        self.dense3_classification  = tf.keras.layers.Dense(4)

        self.loss_list = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


        self.projector = ProjectorBlock(256)
        self.attn1 = AttentionBlock()
        self.attn2 = AttentionBlock()
        self.attn3 = AttentionBlock()
        

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        p1 = self.maxpool(conv1)
        conv2 = self.conv2(p1)
        p2 = self.maxpool(conv2)
        conv3 = self.conv3(p2)
        p3 = self.maxpool(conv3)
        # att = self.attention(p3)
        # att_output = tf.keras.layers.multiply([p3, att])
        conv4 = self.conv4(p3)
        p4 = self.maxpool(conv4)
        # flatten = self.flatten(p4)
        # self.attentionlayer.upfactor = p2.shape[1] // p4.shape[1]

        # attention_output = self.attentionlayer(p2, p4)
        # attention_output = tf.reshape(attention_output, shape=(attention_output.shape[0], -1))
        # d1 = self.dense1_classification(attention_output)


        c1, g1 = self.attn1(self.projector(p1), p4)
        c2, g2 = self.attn2(p2, p4)
        c3, g3 = self.attn3(p3, p4)
        g = tf.concat([g1, g2, g3], axis=1)
    
        d1 = self.dense1_classification(g)
        d2 = self.dense2_classification(d1)
        d3 = self.dense3_classification(d2)
        return d3
    
    def loss(self, logits, labels):
        # class_weights = tf.constant([[6.749, 6.443, 11.781, 1.628]])
        class_weights = tf.constant([[1, 1, 1.2, 0.5]])
        # print(type(class_weights))
        # print(type(labels))
        # weights = tf.constant([class_weights[i] for i in labels])
        weight_per_label = tf.transpose(tf.matmul(labels, tf.transpose(class_weights)))

        # loss = tf.nn.softmax_cross_entropy_with_logits(labels,logits)

        loss = weight_per_label * tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        # weighted_loss = loss * weights
        return tf.reduce_mean(loss)
    
    def accuracy(self, logits, labels):
        
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
