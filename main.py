from __future__ import absolute_import
from preprocess import get_data
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm
from model import Model

NUM_EPOCHS = 50
BATCH_SIZE = 128


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

def train(model, train_inputs, train_labels):
        indices = tf.range(start=0, limit=len(train_labels))
        shuffled_indices = tf.random.shuffle(indices)
        train_inputs = tf.gather(train_inputs, shuffled_indices)
        train_labels = tf.gather(train_labels, shuffled_indices)
    
        combined_data_set = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
        set_of_batches = combined_data_set.batch(BATCH_SIZE)
       
        # up_indices = []
        # for i in range(len(train_labels)):
        #     if np.all(train_labels[i],np.array([0,0,1,0])):
        #         up_indices.append[i]
        
        # up_images = tf.gather(train_inputs,up_indices)
        # count =0 
        # for u in up_images:
        #     u_augment = tf.image.stateless_random_flip_left_right(u)
        #     count+=1
        #     train_inputs[count] = u_augment
        #     train_labels[count] = [0,0,1,0]
           


        # batches_accuracies = []
        # batches_losses = []
        accuracies = []
        count = 1

        for batch_inputs, batch_labels in set_of_batches:
            with tf.GradientTape() as tape:
                logits = model(batch_inputs, training=True)
                loss = model.loss(logits, batch_labels)
                model.loss_list.append(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            accuracy = model.accuracy(logits ,batch_labels)
            accuracies.append(accuracy)
            # batches_losses.append(loss)
            print("batch", count, "loss", loss.numpy(), "acc", accuracy.numpy())
            count += 1
        #visualize_loss(model.loss_list)
        mean_loss = tf.math.reduce_mean(model.loss_list)
        mean_accuracy = tf.math.reduce_mean(accuracies)
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
    dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))
    dataset = dataset.batch(64)

    accuracies = []
    losses = []
    for batch_inputs, batch_labels in dataset:
        logits = model(batch_inputs, training=False)
        loss = model.loss(logits, batch_labels)
        accuracy = model.accuracy(logits, batch_labels)
        accuracies.append(accuracy)
        losses.append(loss)
    
    accuracy = tf.reduce_mean(accuracies)
    loss = tf.reduce_mean(losses)

    return loss, accuracy

def main():
    '''
    Read in the dataset, initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs.
    Print the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    :return: None
    '''
    DATA_FILE = 'data/data.p'

    train_inputs, train_labels, test_inputs, test_labels = get_data(DATA_FILE)
    input_size = train_inputs.shape
    model = Model(input_size, NUM_EPOCHS, 4)
    pbar = tqdm(range(model.num_epoch))
    epoch_loss = []
    for e in range(model.num_epoch):
        loss, acc = train(model, train_inputs, train_labels)
        epoch_loss.append(loss)
        pbar.set_description(f'Epoch {e+1}/{model.num_epoch}: Loss {loss}, Accuracy {acc}\n')
        
    visualize_loss(epoch_loss)


    result_loss, result_acc = test(model, test_inputs, test_labels)
    print("Testing Performance (Loss): ", result_loss.numpy(), "(Accuracy)", result_acc.numpy())
    return


if __name__ == '__main__':
    main()
