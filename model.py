import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import math


class AttentionMap(tf.keras.layers.Layer):
    def __init__(self, fixed_size, **kwargs):
        super(AttentionMap, self).__init__(**kwargs)
        self.fixed_size = fixed_size
        self.K = tf.keras.layers.Dense(fixed_size)
        self.Q = tf.keras.layers.Dense(fixed_size)
        self.V = tf.keras.layers.Dense(fixed_size)

    def call(self,conv_local,conv_global):
        self.K = self.K(conv_local)
        self.V = self.V(conv_local) # not sure if we need this 
        self.Q = self.Q(conv_global)

        weighted_sum = tf.matmul(self.Q,self.K,transpose_b=True)  
        d_k = self.K.shape[2]
        weighted_sum= weighted_sum/tf.sqrt(float(d_k))

        attention_matrix = tf.nn.softmax(weighted_sum)
        attention_map_output = tf.matmul(attention_map_output,self.V)
        return attention_map_output




class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self,upfactor, fixed_size, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.upfactor = upfactor
        self.w_local = tf.keras.layers.Conv2D()
        self.w_global = tf.keras.layers.Conv2D()
        self.attention_module = AttentionMap(fixed_size=fixed_size)
        self.final_conv = tf.keras.layers.Conv2D()
        self.normalize = tf.keras.layers.Softmax()


 
    def call(self, x_local, x_global):
        conv_local = self.w_local(x_local)
        conv_global = self.w_global(x_global)

        #Interpolation -- needs work
        if self.upfactor > 1:
              interpolate_reshape = tf.keras.layers.Reshape(target_shape=(conv_local.shape))
              conv_global = interpolate_reshape(conv_global)
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

        attention_map_output = self.attention_module((conv_local,conv_global))
        attention_layer_output = self.final_conv(attention_map_output)
        return attention_layer_output




class OverallModel(tf.keras.Model):
    def __init__(self, input_size,labels, fixed_size):
        self.filters_1 = tf.Variable(tf.random.truncated_normal([2, 2, input_size[3], fixed_size],dtype=tf.float32,
    stddev=1e-1))
        
        self.conv1 = tf.keras.layers.Conv2D(self.filters_1,fixed_size, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(self.filters_1,fixed_size,padding="same")
        self.conv3 = tf.keras.layers.Conv2D(self.filters_1,fixed_size,padding="same")
        self.conv4 = tf.keras.layers.Conv2D(self.filters_1,fixed_size)
        self.attentionlayer = AttentionBlock(upfactor=2,fixed_size=fixed_size)
        self.dense1_classification  = tf.keras.layers.Dense(fixed_size)
        self.leaky_relu = tf.keras.layers.LeakyReLU(0.3)
        self.dense2_classification  = tf.keras.layers.Dense(fixed_size)
        self.dense3_classification  = tf.keras.layers.Dense(labels.shape[0])

    def call(self, inputs):
        c1 = self.conv1(c1)
        r1 = self.leaky_relu(c1)
        p1 = tf.nn.max_pool(r1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        c2 = self.conv2(p1)
        r2 = self.leaky_relu(c2)
        p2 = tf.nn.max_pool(r2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        c3 = self.conv2(p2)
        r3 = self.leaky_relu(c3)
        p3 = tf.nn.max_pool(r3,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

        attention_output = self.attentionlayer(p3)
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
    
    def train(model, train_inputs, train_labels):
        indices = tf.range(start=0, limit=len(train_labels))
        shuffled_indices = tf.random.shuffle(indices)
        train_inputs = tf.gather(train_inputs, shuffled_indices)
        train_labels = tf.gather(train_labels, shuffled_indices)
    
        combined_data_set = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
        set_of_batches = combined_data_set.batch(model.batch_size)

        batches_accuracies = []

        for batch in set_of_batches:
            batch_inputs, batch_labels = batch
            with tf.GradientTape() as tape:
                pred = model.call(batch_inputs, False)
                pred.shape
                loss = model.loss(pred, batch_labels)
                model.loss_list.append(loss)
            optimizer = tf.keras.optimizers.legacy.Adam(model.learning_rate)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            accuracy = model.accuracy(pred,batch_labels)
            batches_accuracies.append(accuracy)
        
        mean_accuracy = np.mean(batches_accuracies)

        return mean_accuracy


    



        

       
