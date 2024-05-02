import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

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
        # self.conv = tf.keras.layers.Conv2D(filters=out_features, kernel_size=4, padding="SAME")
        
    def call(self, inputs):
        resized_image = tf.image.resize(inputs, size=(self.out_size, self.out_size))
        return resized_image
        # return self.conv(resized_image)


class Model(tf.keras.Model):
    def __init__(self, input_size, num_epoch, kernel_size):
        super(Model, self).__init__()
        
        self.num_epoch = num_epoch
        self.conv1 = tf.keras.layers.Conv2D(filters=32, strides=(2, 2), kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv2 = tf.keras.layers.Conv2D(filters=64, strides=(2, 2), kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv4 = tf.keras.layers.Conv2D(filters=128,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME")

        self.dense1_classification  = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())
        self.dense2_classification  = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU())
        self.dense3_classification  = tf.keras.layers.Dense(4)

        self.loss_list = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.dropout_rate = 0.15
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate) 

        self.projector1 = ProjectorBlock(25)
        self.projector2 = ProjectorBlock(7)
        self.projector3 = ProjectorBlock(4)
        self.attn1 = AttentionBlock()
        self.attn2 = AttentionBlock()
        self.attn3 = AttentionBlock()
        

    def call(self, inputs, training: bool):
        conv1 = self.conv1(inputs)
        p1 = self.maxpool(conv1)
        if training: p1 = self.dropout(p1)
        else: p1 = p1 * (1 - self.dropout_rate)
        # print(p1.shape) # (256, 25, 25, 32)
        conv2 = self.conv2(p1)
        p2 = self.maxpool(conv2)
        if training: p2 = self.dropout(p2)
        else: p2 = p2 * (1 - self.dropout_rate)
        # print(p2.shape) # (256, 7, 7, 64)
        conv3 = self.conv3(p2)
        p3 = self.maxpool(conv3)
        if training: p3 = self.dropout(p3)
        else: p3 = p3 * (1 - self.dropout_rate)
        # print(p3.shape) # (256, 4, 4, 128)
        conv4 = self.conv4(p3)
        p4 = self.maxpool(conv4)
        if training: p4 = self.dropout(p4)
        else: p4 = p4 * (1 - self.dropout_rate)
        # print(p4.shape) # (256, 2, 2, 128)

        c1, g1 = self.attn1(p1, self.projector1(p4))
        c2, g2 = self.attn2(p2, self.projector2(p4))
        c3, g3 = self.attn3(p3, self.projector3(p4))
        g = tf.concat([g1, g2, g3], axis=1)
    
        d1 = self.dense1_classification(g)
        d2 = self.dense2_classification(d1)
        d3 = self.dense3_classification(d2)
        return d3
    
    def loss(self, logits, labels):
        # class_weights = tf.constant([[6.749, 6.443, 11.781, 1.628]])
        # class_weights = tf.constant([[0.25, 0.24, 0.435, 0.065]])
        # class_weights = tf.constant([[1, 1, 1.2, 0.9]])
        
        # weight_per_label = tf.transpose(tf.matmul(labels, tf.transpose(class_weights)))
        # loss = weight_per_label * tf.nn.softmax_cross_entropy_with_logits(labels, logits)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels,logits)
        return tf.reduce_mean(loss)
    
    def accuracy(self, logits, labels):
        
        x = tf.nn.softmax(logits)
        correct_predictions = tf.equal(tf.argmax(x, 1), tf.argmax(labels, 1))
        # preds = tf.argmax(x, 1).numpy()
        # print("blink", len([pred for pred in preds if pred == 3]))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    

    def visualize_cnn_layer(self, img, layer_name, nrows, ncols, figsize, view_img=True):
        '''
        Not working yet
        '''
        img = np.array(img)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        curr_layer = self.get_layer(layer_name).output
        slice_model  = tf.keras.Model(inputs=self.inputs, outputs=curr_layer)
        slice_output = slice_model.predict(img[None,:,:,:])

        for row in range(nrows):
            for col in range(ncols):
                idx = row * ncols + col
                curr_ax = axes[row, col]
                out = slice_output[0,:,:,idx].astype(np.uint8)
                out = Image.fromarray(out)
                out = out.resize(img.shape[:-1], resample=Image.BOX)
                curr_ax.imshow(out)
                if view_img:
                    curr_ax.imshow(img, alpha=0.3)

        return fig, axes