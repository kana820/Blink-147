from __future__ import absolute_import
from preprocess import get_data
import os
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import math
from tqdm import tqdm
from model import Model
from matplotlib import pyplot as plt
from PIL import Image
from model import Model
from sklearn.metrics import confusion_matrix
import seaborn as sns

NUM_EPOCHS = 40
BATCH_SIZE = 128

def filter_by_class(dataset, class_index):
    return dataset.filter(lambda x, y: tf.argmax(y) == class_index)

def weighted_sampling(dataset):

    class_names = ["left", "right", "up", "blink"]

    class_datasets = [filter_by_class(dataset, i) for i in range(len(class_names))]

    for i, class_dataset in enumerate(class_datasets):
        count = 0
        for _, _ in class_dataset:
            count += 1
        print(f"Number of examples in class {class_names[i]}:", count)

    left_class = class_datasets[0]
    right_class = class_datasets[1]
    up_class = class_datasets[2]
    blink_class = class_datasets[3]

    all_data_weighted = tf.data.Dataset.sample_from_datasets(
        [left_class.repeat(900 // 222), right_class.repeat(math.ceil(900 / 245)), up_class.repeat(900 // 126), blink_class], weights=[.25, .25, .25, .25])
    
    all_data_weighted = all_data_weighted.shuffle(5000)
    
    # all_data_weighted = tf.data.Dataset.sample_from_datasets(
    #     [left_class, right_class, up_class, blink_class], weights=[.25, .25, .25, .25], stop_on_empty_dataset=True)
    
    # xs = []
    # for x in list(all_data_weighted.as_numpy_iterator()):
    #     if x[1][0] == 1:
    #         xs.append(0)
    #     elif x[1][1] == 1:
    #         xs.append(1)
    #     elif x[1][2] == 1:
    #         xs.append(2)
    #     elif x[1][3] == 1:
    #         xs.append(3)

    # xs = np.array(xs)
    # # print(xs)

    # np.bincount(xs)

    return all_data_weighted

def train(model, train_inputs, train_labels): 

        indices = tf.range(start=0, limit=len(train_labels))
        shuffled_indices = tf.random.shuffle(indices)
        train_inputs = tf.gather(train_inputs, shuffled_indices)
        train_labels = tf.gather(train_labels, shuffled_indices)

        # train_inputs = tf.image.random_flip_left_right(train_inputs)
    
        combined_data_set = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))

        # combined_data_set = weighted_sampling(combined_data_set)

        set_of_batches = combined_data_set.batch(BATCH_SIZE)

        epoch_loss = []
        accuracies = []
        count = 1

        # image_10_array = train_inputs[10].numpy() * 255
        # image_10_array = np.array(image_10_array, dtype=np.uint8)
        # image_10 = Image.fromarray(image_10_array)
        # plt.imshow(image_10)
        # plt.title('original')
        # plt.show()

        for batch_inputs, batch_labels in set_of_batches:
            with tf.GradientTape() as tape:
                logits = model(batch_inputs, training=True)
                loss = model.loss(logits, batch_labels)
                model.loss_list.append(loss)
                epoch_loss.append(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            accuracy = model.accuracy(logits, batch_labels)
            model.acc_list.append(accuracy)
            accuracies.append(accuracy)
            # batches_losses.append(loss)
            print("batch", count, "loss", loss.numpy(), "acc", accuracy.numpy())
            count += 1
        
        mean_loss = tf.math.reduce_mean(epoch_loss)
        model.epoch_loss.append(mean_loss)
        mean_accuracy = tf.math.reduce_mean(accuracies)
        model.epoch_acc.append(mean_accuracy)
        return mean_loss, mean_accuracy

def test(model, test_inputs, test_labels, vis_cnn):
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
    test_pred = []
    for batch_inputs, batch_labels in dataset:
        logits = model(batch_inputs, training=False)
        loss = model.loss(logits, batch_labels)
        accuracy = model.accuracy(logits, batch_labels)
        accuracies.append(accuracy)
        losses.append(loss)
        test_pred = np.append(test_pred, tf.argmax(logits, 1).numpy())
    
    accuracy = tf.reduce_mean(accuracies)
    model.test_acc.append(accuracy)
    loss = tf.reduce_mean(losses)

    if vis_cnn:
        test_true = tf.argmax(test_labels, 1)
        cm = confusion_matrix(test_true, test_pred)
        sns.heatmap(cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=['left', 'right', 'up', 'blink'], yticklabels=['left', 'right', 'up', 'blink'])
        plt.xlabel('Prediction',fontsize=12)
        plt.ylabel('Actual',fontsize=12)
        plt.title('Confusion Matrix',fontsize=16)
        plt.show()

    return loss, accuracy

def visualize_loss(model): 
    ratio = len(model.loss_list)/len(model.epoch_loss)
    x = [i/ratio for i in range(len(model.loss_list))]
    x_epoch = [i+1 for i in range(len(model.epoch_loss))]
    plt.plot(x, model.loss_list, color="lightblue") # all loss
    plt.plot(x_epoch, model.epoch_loss, color="b") # epoch loss
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() 

def visualize_acc(model): 
    # ratio = len(model.acc_list)/len(model.epoch_acc)
    # x = [i/ratio for i in range(len(model.acc_list))]
    x_epoch = [i+1 for i in range(len(model.epoch_acc))]
    # plt.plot(x, model.acc_list, color="lightcoral") # all acc
    plt.plot(x_epoch, model.epoch_acc, color="red") # epoch acc
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show() 

def visualize_train_test_acc(model):
    x = [i+1 for i in range(len(model.epoch_acc))]
    x_2 = [(i+1)*5 for i in range(len(model.test_acc))]
    plt.plot(x, model.epoch_acc, color="red") # train acc
    plt.plot(x_2, model.test_acc, color="b")
    plt.title('Train v Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show() 


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
    for e in range(model.num_epoch):
        loss, acc = train(model, train_inputs, train_labels)
        pbar.set_description(f'Epoch {e+1}/{model.num_epoch}: Loss {loss}, Accuracy {acc}\n')
        print(e)
        if ((e+1) % 5==0):
            test_loss, test_acc = test(model, test_inputs, test_labels, vis_cnn=False)
            print("Testing Performance Epoch", e, "Loss: ", test_loss.numpy(), "Accuracy: ", test_acc.numpy())
    
    with Image.open('data/train_images/Alecos_Markides_0001.jpg') as img:
        model.visualize_cnn_layer(img, 4, 4, (15, 15), view_img=True)
        plt.show()
        # model.visualize_attention(img)

    visualize_train_test_acc(model)

    model.model().summary()
    result_loss, result_acc = test(model, test_inputs, test_labels, vis_cnn=True)
    print("Final Testing Performance (Loss): ", result_loss.numpy(), "(Accuracy)", result_acc.numpy())
    visualize_loss(model)
    visualize_acc(model) 


    return


if __name__ == '__main__':
    main()
