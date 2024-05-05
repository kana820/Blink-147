import numpy as np
import pickle
import random
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt

def load_data(data_folder):
    '''
    preprocess the data to create the data.p file of images and labels

    :param data_folder: folder containing data to preprocess
	:return: dictionary of image and label arrays
    '''
    data_label_path = f'{data_folder}/train_labels.txt'

    with open(data_label_path) as file:
        examples = file.read().splitlines()
   
    # map each image name to its label
    image_names_to_labels = {}
    for example in examples:
        img_name, label = example.split(',')
        image_names_to_labels[img_name] = image_names_to_labels.get(img_name, []) + [label]

    img_file_path = f'{data_folder}/dataset'

    # Shuffle the order and retrieve the image data
    image_names = list(image_names_to_labels.keys())
    random.shuffle(image_names)
    images = get_image_from_dir(img_file_path, image_names)    

    # returns labels for all images in the array
    def get_all_labels(image_names, image_names_to_labels):
        to_return = []
        for image in image_names:
            label = image_names_to_labels[image]
            to_return.append(label)
        return to_return
   
    labels = get_all_labels(image_names, image_names_to_labels)

    return dict(
        images          = images,
        labels          = labels,        
    )

def get_image_from_dir(data_folder, image_names):
    '''Retrieves image from data folder.
    
    :param data_folder: folder containing data to preprocess, image_names: list of names of images
	:return: list of images
    '''
    images = []
    for image in image_names:
        image_path = os.path.join(data_folder, image)
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                img_array = np.array(img)
                images.append(img_array)
        else:
            print(image)
    return images

def create_pickle(data_folder):
    '''Creates pickle file called data.p to dump data.
    
    :param data_folder: folder containing data to preprocess
    '''
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')

def unpickle(file):
	"""
	Opens "pickled" object.

	:param file: file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def get_data(file_path):
    """
    Unpickles data, normalizes images, and one-hot encodes labels.

    :param file_path: file path for images inputs and labels
    :return: tuple of train inputs, train labels, test inputs, test labels
    """
    unpickled_file = unpickle(file_path)
    inputs = unpickled_file.get('images')
    labels = unpickled_file.get('labels')

    # Split to training and testing
    train_len = int(len(labels) * 0.75)
    train_inputs = inputs[:train_len]
    train_labels = labels[:train_len]
    test_inputs = inputs[train_len:]
    test_labels = labels[train_len:]
    
    def normalize_images(images):
        images = np.array(images)
        images = images / 255
        return images # shape (num_examples, 100, 100, 3)
    
    train_inputs = normalize_images(train_inputs)
    test_inputs = normalize_images(test_inputs)

    # Convert labels in string to ints
    # Left=0, Right=1, Up=2, Blink=3
    label_map = {"left": 0, "right": 1, "up": 2, "blink": 3}
    train_labels = [label_map[label[0]] for label in train_labels]
    test_labels = [label_map[label[0]] for label in test_labels]

    def print_data_breakdown(labels):
        ''' Keeps count of data breakdown.

        :param labels: list of image labels to count
        '''
        left = 0
        right = 0
        up = 0
        blink = 0
        for label in labels:
            if label == 0: left += 1
            if label == 1: right += 1
            if label == 2: up += 1
            if label == 3: blink += 1
        print("left", left, ", right", right, ", up", up, ", blink", blink)

    print("num of train samples", len(train_inputs))
    print_data_breakdown(train_labels)
    print("num of test sample", len(test_inputs))
    print_data_breakdown(test_labels)

    # Turn your labels into one-hot vectors
    train_labels = tf.one_hot(train_labels, 4)
    test_labels = tf.one_hot(test_labels, 4)
    return train_inputs, train_labels, test_inputs, test_labels

if __name__ == '__main__':
    # make a pickle file from the dataset
    # data_folder = 'data'
    # create_pickle(data_folder)
    pass