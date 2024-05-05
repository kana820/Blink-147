import numpy as np
import pickle
import random
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt

def load_data(data_folder):
    '''
    Preprocesses data to create the augmented_data.p file.

    - initializes dictionary with image name/label key/value pairs
    - shuffles images
    - retrieves labels in shuffled order
    - splits data 75/25 for training and testing
    - separates training data by label
    - downsamples "blink" and augments all other labels by flipping and cropping
    - retrieves image inputs

    :param data_folder: folder containing data to preprocess
	:return: dictionary of train and test images and labels arrays
    '''
    image_label_path = f'{data_folder}/train_labels.txt'
    image_folder_path = f'{data_folder}/dataset'

    with open(image_label_path) as file:
        examples = file.read().splitlines()
    
    # inputs images/labels as K/V in dict
    image_names_to_labels = {}
    for example in examples:
        img_name, label = example.split(',')
        image_names_to_labels[img_name] = label

    # shuffles images
    shuffled_image_names = list(image_names_to_labels.keys())
    random.shuffle(shuffled_image_names)

    # returns labels for images in order
    def get_all_labels(image_names, image_names_to_labels: dict):
        to_return = []
        for image in image_names:
            label = image_names_to_labels[image]
            to_return.append(label)
        return to_return
   
    labels = get_all_labels(shuffled_image_names, image_names_to_labels)

    # splits training and testing
    train_len = int(len(labels) * 0.75)
    train_image_names = shuffled_image_names[:train_len]
    test_image_names = shuffled_image_names[train_len:]
    test_labels = labels[train_len:]

    # separates the training dataset for each label
    left_image_names = [train_image_name for train_image_name in train_image_names
                            if image_names_to_labels[train_image_name] == 'left']
    right_image_names = [train_image_name for train_image_name in train_image_names
                        if image_names_to_labels[train_image_name] == 'right']
    up_image_names = [train_image_name for train_image_name in train_image_names
                    if image_names_to_labels[train_image_name] == 'up']
    blink_image_names = [train_image_name for train_image_name in train_image_names
                        if image_names_to_labels[train_image_name] == 'blink']
    
    # upsampling by data augmentation
    # for 'left'
    left_images = get_image_from_dir(image_folder_path, left_image_names)
    left_labels = ['left'] * len(left_images)
    left_images_flipped_contrast = tf.image.adjust_contrast(tf.image.flip_left_right(left_images), 2.)
    left_labels_flipped_contrast = ['right'] * len(left_images)  # labels turned to RIGHT

    # for 'right'
    right_images = get_image_from_dir(image_folder_path, right_image_names)
    right_labels = ['right'] * len(right_images)
    right_images_flipped_contrast = tf.image.adjust_contrast(tf.image.flip_left_right(right_images), 2.)
    right_labels_flipped_contrast = ['left'] * len(right_images)  # labels turned to LEFT

    # for 'up'
    up_images = get_image_from_dir(image_folder_path, up_image_names)
    up_labels = ['up'] * len(up_images)
    up_images_flipped_contrast = tf.image.adjust_contrast(tf.image.flip_left_right(up_images), 2.)
    up_labels_flipped_contrast = ['up'] * len(up_images)
    up_images_cropped = tf.image.adjust_saturation(tf.image.central_crop(up_images, 0.85), 0.3)
    up_images_cropped_resized = tf.cast(tf.image.resize(up_images_cropped, [100, 100]), dtype=tf.int32)  # keep the original size
    up_labels_cropped_resized = ['up'] * len(up_images)
    
    # downsampling for 'blink'
    blink_images_downsampled = get_image_from_dir(image_folder_path, blink_image_names[:500])
    blink_lables_downsampled = ['blink'] * 500

    train_images = tf.concat([left_images, left_images_flipped_contrast, right_images,
                              right_images_flipped_contrast, up_images, up_images_flipped_contrast,
                               up_images_cropped_resized, blink_images_downsampled], axis=0)
    train_labels = left_labels + left_labels_flipped_contrast + right_labels + right_labels_flipped_contrast \
                       + up_labels + up_labels_flipped_contrast + up_labels_cropped_resized + blink_lables_downsampled
    
    assert(train_images.shape[0] == len(train_labels))
    
    test_images = get_image_from_dir(image_folder_path, test_image_names)

    return dict(
        test_labels = test_labels,
        test_images = test_images,
        train_labels = train_labels,
        train_images = train_images
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
    '''Creates pickle file to dump data.
    
    :param data_folder: folder containing data to preprocess
    '''
    with open(f'{data_folder}/data_augmented.p', 'wb') as pickle_file:
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
    # unpickles training and testing data
    unpickled_file = unpickle(file_path)
    train_inputs = unpickled_file.get('train_images')
    train_labels = unpickled_file.get('train_labels')
    test_inputs = unpickled_file.get('test_images')
    test_labels = unpickled_file.get('test_labels')
    
    def normalize_images(images):
        '''Normalizes images to scale pixel values between 0 and 1.

        :param images: image input from data
        :return: normalized image with scaled pixels
        '''
        images = np.array(images)
        images = images / 255
        return images
    
    # normalize images
    train_inputs = normalize_images(train_inputs)
    test_inputs = normalize_images(test_inputs)

    # convert labels in string to ints
    # Left = 0, Right = 1, Up = 2, Blink = 3
    label_map = {"left": 0, "right": 1, "up": 2, "blink": 3}
    train_labels = [label_map[label] for label in train_labels]
    test_labels = [label_map[label] for label in test_labels]

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

    # one-hot encodes labels
    train_labels = tf.one_hot(train_labels, 4)
    test_labels = tf.one_hot(test_labels, 4)
    return train_inputs, train_labels, test_inputs, test_labels

if __name__ == '__main__':
    # make a pickle file from the dataset
    data_folder = 'data'
    # create_pickle(data_folder)
    # load_data(data_folder)