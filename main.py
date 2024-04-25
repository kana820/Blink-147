from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm
from model import Model

NUM_EPOCHS = 10

def train(model, train_inputs, train_labels):
        indices = tf.range(start=0, limit=len(train_labels))
        shuffled_indices = tf.random.shuffle(indices)
        train_inputs = tf.gather(train_inputs, shuffled_indices)
        train_labels = tf.gather(train_labels, shuffled_indices)
    
        combined_data_set = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
        set_of_batches = combined_data_set.batch(model.batch_size)

        batches_accuracies = []
        batches_losses = []

        for batch in set_of_batches:
            batch_inputs, batch_labels = batch
            with tf.GradientTape() as tape:
                pred = model.call(batch_inputs, False)
                loss = model.loss(pred, batch_labels)
                model.loss_list.append(loss)
            optimizer = tf.keras.optimizers.legacy.Adam(model.learning_rate)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            accuracy = model.accuracy(pred,batch_labels)
            batches_accuracies.append(accuracy)
            batches_losses.appen(loss)
        
        mean_loss = tf.math.reduce_mean(batches_losses)
        mean_accuracy = tf.math.reduce_mean(batches_accuracies)
        return mean_loss, mean_accuracy

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    pred = model(test_inputs, is_testing=True)  # Forward pass
    loss = model.loss(pred, test_labels)
    accuracy = model.accuracy(pred,test_labels)
    return loss, accuracy

def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    # Read in CIFAR10 data (limited to two classes, cats and dogs)
    train_inputs, train_labels = get_data(AUTOGRADER_TRAIN_FILE, 3, 5)
    test_inputs, test_labels = get_data(AUTOGRADER_TEST_FILE, 3, 5)

    model = Model()
    pbar = tqdm(range(model.num_epoch))
    for e in range(model.num_epoch):
        loss, acc = train(model, train_inputs, train_labels)
        pbar.set_description(f'Epoch {e+1}/{model.num_epoch}: Loss {loss}, Accuracy {acc}\n')

    result_loss, result_acc = test(model, test_inputs, test_labels)
    print("Testing Performance (Loss): ", result_loss.numpy(), "(Accuracy)", result_acc.numpy())
    return


if __name__ == '__main__':
    main()
