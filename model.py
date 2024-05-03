import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from PIL import Image

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='valid', use_bias=False)

    def call(self, l, g):
        # N, W, H, C = tf.shape(l)

        N, W, H, C = tf.split(tf.shape(l), num_or_size_splits=4)
        N = tf.squeeze(N, axis=0)
        W = tf.squeeze(W, axis=0)
        H = tf.squeeze(H, axis=0)
        C = tf.squeeze(C, axis=0)

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

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU(), input_shape=(100,100,3))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv3 = tf.keras.layers.Conv2D(filters=64,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.conv4 = tf.keras.layers.Conv2D(filters=64,kernel_size=kernel_size, padding="SAME", activation=tf.keras.layers.LeakyReLU())
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="SAME")

        # self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)

        # self.flatten = tf.keras.layers.Flatten()

        self.dense1_classification  = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())
        self.dense2_classification  = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU())
        self.dense3_classification  = tf.keras.layers.Dense(4)

        self.loss_list = []
        self.epoch_loss = []
        self.acc_list = []
        self.epoch_acc = []
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        # self.dropout_rate = 0.15
        # self.dropout = tf.keras.layers.Dropout(self.dropout_rate)


        # self.projector = ProjectorBlock(7)
        self.projector1 = ProjectorBlock(50)
        self.projector2 = ProjectorBlock(25)
        self.projector3 = ProjectorBlock(13)
        self.attn1 = AttentionBlock()
        self.attn2 = AttentionBlock()
        self.attn3 = AttentionBlock()
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        

    def call(self, inputs, training: bool):
        conv1 = self.conv1(inputs)
        p1 = self.maxpool(conv1)
        # if training: p1 = self.dropout(p1)
        # else: p1 = p1 * (1 - self.dropout_rate)

        conv2 = self.conv2(p1)
        p2 = self.maxpool(conv2)
        # if training: p2 = self.dropout(p2)
        # else: p2 = p2 * (1 - self.dropout_rate)
        # print(p2.shape) # (256, 7, 7, 64)
        conv3 = self.conv3(p2)
        p3 = self.maxpool(conv3)
        # if training: p3 = self.dropout(p3)
        # else: p3 = p3 * (1 - self.dropout_rate)
        # print(p3.shape)
        conv4 = self.conv4(p3)
        p4 = self.maxpool(conv4)
        # if training: p4 = self.dropout(p4)
        # else: p4 = p4 * (1 - self.dropout_rate)
        # flatten = self.flatten(p4)

        c1, g1 = self.attn1(p1, self.projector1(p4))
        c2, g2 = self.attn2(p2, self.projector2(p4))
        c3, g3 = self.attn3(p3, self.projector3(p4))
        g = tf.concat([g1, g2, g3], axis=1)

        g = self.layer_norm(g)
    
        d1 = self.dense1_classification(g)
        d2 = self.dense2_classification(d1)
        d3 = self.dense3_classification(d2)
        return d3
    
    def loss(self, logits, labels):
        # class_weights = tf.constant([[6.749, 6.443, 11.781, 1.628]])

        # weights = tf.constant([class_weights[i] for i in labels])
        # weight_per_label = tf.transpose(tf.matmul(labels, tf.transpose(class_weights)))

        loss = tf.nn.softmax_cross_entropy_with_logits(labels,logits)

        # loss = weight_per_label * tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        # weighted_loss = loss * weights
        return tf.reduce_mean(loss)
    
    def accuracy(self, logits, labels):
        x = tf.nn.softmax(logits)
        correct_predictions = tf.equal(tf.argmax(x, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    def model(self):
        x = tf.keras.layers.Input(shape=(100, 100, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x, training=True))
    
    def visualize_cnn_layer(self, img, nrows, ncols, figsize, view_img=True):
        '''
        Not working yet
        '''
        img = np.array(img) / 255
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        img_out = self.conv1(img[None,:,:,:])
        # slice_model  = tf.keras.Model(inputs=self.inputs, outputs=curr_layer)
        slice_output = self.conv2(img_out)

        for row in range(nrows):
            for col in range(ncols):
                idx = row * ncols + col
                curr_ax = axes[row, col]
                out = np.array(slice_output[0,:,:,idx].numpy() * 255, dtype=np.uint8)
                out = Image.fromarray(out)
                out = out.resize((100, 100), resample=Image.BOX)
                curr_ax.imshow(out)
                if view_img:
                    curr_ax.imshow(img, alpha=0.3)

        return fig, axes
    
    def visualize_attention(self, img):
        # image
        img = np.array(img)  / 255

        conv1_out = self.maxpool(self.conv1(img[None,:,:,:]))
        conv2_out = self.maxpool(self.conv2(conv1_out))
        conv3_out = self.maxpool(self.conv3(conv2_out))
        conv4_out = self.maxpool(self.conv4(conv3_out))

        print(conv4_out.shape)

        c1, g1 = self.attn1(conv1_out, self.projector1(conv4_out))
        c2, g2 = self.attn1(conv2_out, self.projector2(conv4_out))
        c3, g3 = self.attn1(conv3_out, self.projector3(conv4_out))

        out1 = np.squeeze(np.array(c1.numpy() * 255).astype(np.uint8))
        out1 = Image.fromarray(out1)
        out1 = out1.resize((100, 100), resample=Image.BOX)

        out2 = np.squeeze(np.array(c2.numpy() * 255).astype(np.uint8))
        out2 = Image.fromarray(out2)
        out2 = out2.resize((100, 100), resample=Image.BOX)

        out3 = np.squeeze(np.array(c3.numpy() * 255).astype(np.uint8))
        out3 = Image.fromarray(out3)
        out3 = out2.resize((100, 100), resample=Image.BOX)

        outs = [out1, out2, out3]

        fig, axs = plt.subplots(1, 3)
        for col in range(3):
            ax = axs[col]
            curr_plot = outs[col]
            pcm = ax.imshow(curr_plot)
            ax.imshow(img, alpha=0.85)
            plot_title = "Attention Map " + str(col)
            ax.set_title(plot_title)
        fig.colorbar(pcm, ax=axs[:], shrink=0.4)
        plt.show()