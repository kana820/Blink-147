import os
import tensorflow as tf
import numpy as np
import random
import math

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.w_local = tf.keras.layers.Conv2D()
        self.w_global = tf.keras.layers.Conv2D()
        self.final_conv = tf.keras.layers.Conv2D()
        self.normalize = tf.keras.layers.Softmax()
 
    def call(self, x_local, x_global):
        conv_local = self.w_local(x_local)
        conv_global = self.w_global(x_global)

        interpolate = interpolate # bro idk what to do here

        

        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
