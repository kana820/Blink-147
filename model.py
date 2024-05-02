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
        c = self.op(tf.concat([l, g], axis=-1))  # batch_sizexWxHx1
        c = tf.squeeze(c, axis=-1)

        if self.normalize_attn:
            a = tf.nn.softmax(tf.reshape(c, [N, -1]), axis=1)  # batch_sizex(W*H)
            a = tf.reshape(a, [N, W, H, 1])  # batch_sizexWxHx1
        else:
            a = tf.sigmoid(c)
            a = tf.expand_dims(a, axis=-1)  # batch_sizexWxHx1

        g = tf.multiply(a, l)
        if self.normalize_attn:
            g = tf.reduce_sum(g, axis=[1, 2])  # batch_sizexC
        else:
            g = tf.reduce_mean(g, axis=[1, 2], keepdims=True)  # batch_sizex1x1xC
            g = tf.squeeze(g, axis=[1, 2])  # batch_sizexC
        
        return tf.expand_dims(c, axis=-1), g

class ProjectorBlock(tf.keras.layers.Layer):
    def __init__(self, out_size):
        super(ProjectorBlock, self).__init__()
        self.out_size = out_size
        
    def call(self, inputs):
        # return self.op(inputs)
        return tf.image.resize(inputs, size=(self.out_size, self.out_size))


class Model(tf.keras.Model):
    def __init__(self, input_size, num_epoch, kernel_size):
        super(Model, self).__init__()
        
        self.num_epoch = num_epoch
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv2 = tf.keras.layers.Conv2D(filters=32,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv3 = tf.keras.layers.Conv2D(filters=32,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv4 = tf.keras.layers.Conv2D(filters=64,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME")

        # self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)

        # self.flatten = tf.keras.layers.Flatten()
        # self.attention = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")
        # Conv2D(1, (1, 1), activation='sigmoid')(x)
        # x = multiply([x, attention])
        # self.attentionlayer = AttentionBlock(kernel_size=kernel_size)

        self.dense1_classification  = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU())
        self.dense2_classification  = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())
        self.dense3_classification  = tf.keras.layers.Dense(4)

        self.loss_list = []
        self.epoch_loss = []
        self.acc_list = []
        self.epoch_acc = []
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)


        self.projector = ProjectorBlock(7)
        self.attn1 = AttentionBlock()
        self.attn2 = AttentionBlock()
        self.attn3 = AttentionBlock()
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        p1 = self.maxpool(conv1)
        conv2 = self.conv2(p1)
        p2 = self.maxpool(conv2)
        conv3 = self.conv3(p2)
        p3 = self.maxpool(conv3)
        conv4 = self.conv4(p3)
        p4 = self.maxpool(conv4)
        # flatten = self.flatten(p4)

        c1, g1 = self.attn1(self.projector(p1), p4)
        c2, g2 = self.attn2(self.projector(p2), p4)
        c3, g3 = self.attn3(self.projector(p3), p4)
        g = tf.concat([g1, g2, g3], axis=1)

        g = self.layer_norm(g)
    
        d1 = self.dense1_classification(g)
        d2 = self.dense2_classification(d1)
        d3 = self.dense3_classification(d2)
        return d3
    
    def loss(self, logits, labels):
        class_weights = tf.constant([[6.749, 6.443, 11.781, 1.628]])
        # class_weights = tf.constant([[1, 1, 1.2, 0.5]])
        # print(type(class_weights))
        # print(type(labels))
        # weights = tf.constant([class_weights[i] for i in labels])
        weight_per_label = tf.transpose(tf.matmul(labels, tf.transpose(class_weights)))

        # loss = tf.nn.softmax_cross_entropy_with_logits(labels,logits)

        loss = weight_per_label * tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        # weighted_loss = loss * weights
        return tf.reduce_mean(loss)
    
    def accuracy(self, logits, labels):
        x = tf.nn.softmax(logits)
        correct_predictions = tf.equal(tf.argmax(x, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
