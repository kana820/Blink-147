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

NUM_EPOCHS = 50
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

        train_inputs = tf.image.random_flip_left_right(train_inputs)
    
        combined_data_set = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))

        combined_data_set = weighted_sampling(combined_data_set)

        set_of_batches = combined_data_set.batch(BATCH_SIZE)
        # set_of_batches = all_data_weighted.batch(BATCH_SIZE)

        # batches_accuracies = []
        # batches_losses = []
        epoch_loss = []
        accuracies = []
        count = 1

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

def visualize_loss(model): 
    ratio = len(model.loss_list)/len(model.epoch_loss)
    x = [i/ratio for i in range(len(model.loss_list))]
    x_epoch = [i+1 for i in range(len(model.epoch_loss))]
    plt.plot(x, model.loss_list, color="lightblue") # all loss
    plt.plot(x_epoch, model.epoch_loss, color="b") # epoch loss
    plt.title('Loss (Attention-Based Model with Weighted Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() 

def visualize_acc(model): 
    ratio = len(model.acc_list)/len(model.epoch_acc)
    x = [i/ratio for i in range(len(model.acc_list))]
    x_epoch = [i+1 for i in range(len(model.epoch_acc))]
    plt.plot(x, model.acc_list, color="lightcoral") # all losses
    plt.plot(x_epoch, model.epoch_acc, color="red") # epoch loss
    plt.title('Accuracy (Attention-Based Model with Weighted Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show() 

# def visualize_feature_maps(model, layer_name, input_image):
#     intermediate_layer_model = tf.keras.Model(inputs=model.inputs,
#                                                outputs=model.get_layer(layer_name).output)
#     intermediate_output = intermediate_layer_model.predict(input_image)
    
#     n_filters = intermediate_output.shape[3]
#     n_columns = 8
#     n_rows = math.ceil(n_filters / n_columns)
    
#     plt.figure(figsize=(20, 10))
#     for i in range(n_filters):
#         plt.subplot(n_rows, n_columns, i+1)
#         plt.imshow(intermediate_output[0, :, :, i], cmap='viridis')
#         plt.axis('off')
#     plt.suptitle(layer_name + ' Feature Maps')
#     plt.show()

def visualize_vgg_layer(img, model, layer_name, nrows, ncols, figsize, view_img=True):
    # img = img.resize((224,224))
    img = np.array(img)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    curr_layer = model.get_layer(layer_name).output
    slice_model  = tf.keras.Model(inputs=model.inputs, outputs=curr_layer)
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

    model.model().summary()
    result_loss, result_acc = test(model, test_inputs, test_labels)
    print("Testing Performance (Loss): ", result_loss.numpy(), "(Accuracy)", result_acc.numpy())
    visualize_loss(model)
    visualize_acc(model)

    img_index = 0  
    input_image = np.expand_dims(train_inputs[img_index], axis=0)  # Assuming you're using train_inputs
    # visualize_feature_maps(model, 'conv2d', input_image)
    # visualize_feature_maps(model, 'conv2d_1', input_image)
    # visualize_feature_maps(model, 'conv2d_2', input_image)
    # visualize_feature_maps(model, 'conv2d_3', input_image)
    # visualize_vgg_layer(input_image, model, 'conv2d', 2, 2, (15,15))
    

    return


if __name__ == '__main__':
    main()
