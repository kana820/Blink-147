import numpy as np
import pickle
import tensorflow as tf
import os

def load_data(data_folder):
    '''
    Method that was used to preprocess the data in the data.p file. You do not need 
    to use this method, nor is this used anywhere in the assignment. This is the method
    that the TAs used to pre-process the Flickr 8k dataset and create the data.p file 
    that is in your assignment folder. 

    Feel free to ignore this, but please read over this if you want a little more clairity 
    on how the images and captions were pre-processed 
    '''
    
    test_fle_path = f'{data_folder}/test_labels.txt'
    train_file_path = f'{data_folder}/train_labels.txt'

    with open(test_fle_path) as file:
        test_examples = file.read().splitlines()[1:]
        
    with open(train_file_path) as file:
        train_examples = file.read().splitlines()[1:]
    
    # map each image name to its captions
    test_image_names_to_labels = {}
    for test_example in test_examples:
        test_img_name, test_label = test_example.split(',', 1)
        test_image_names_to_labels[test_img_name] = test_image_names_to_labels.get(test_img_name, []) + [test_label]
        
    train_image_names_to_labels = {}
    for train_example in train_examples:
        train_img_name, train_label = train_example.split(',', 1)
        train_image_names_to_labels[train_img_name] = train_image_names_to_labels.get(train_img_name, []) + [train_label]

    test_images = list(test_image_names_to_labels.keys())
    train_images = list(train_image_names_to_labels.keys())

    # returns all captions for all images
    def get_all_labels(image_names, image_names_to_labels):
        to_return = []
        for image in image_names:
            label = image_names_to_labels[image]
            to_return.append(label)
        return to_return
    
    test_labels = get_all_labels(test_images, test_image_names_to_labels)
    train_labels = get_all_labels(train_images, train_image_names_to_labels)

    return dict(
        test_labels          = np.array(test_labels),
        test_images            = np.array(test_images),
        train_labels          = np.array(train_labels),
        train_images            = np.array(train_images),
    )


def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def get_data(file_path):
    """
    Given a file path and two target classes, returns an array of
    normalized inputs (images) and an array of labels
    :param file_path: file path for inputs and labels
    :return: normalized NumPy array of inputs and tensor of labels, where 
    inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
    has size (num_examples, num_classes)
    """
    unpickled_file = unpickle(file_path)
    train_inputs = unpickled_file[b'train_images']
    train_labels = unpickled_file[b'train_labels']
    test_inputs = unpickled_file[b'test_images']
    test_labels = unpickled_file[b'test_labels']
    
    def reshape_images(images):
         images = images / 255
         images = tf.reshape(images, (-1, 3, 100, 100))
         images = tf.transpose(images, perm=[0,2,3,1])
         # Now in the shape (num_examples, 100, 100, 3)
         return images
    
    train_inputs = reshape_images(train_inputs)
    test_inputs = reshape_images(test_inputs)
    # Turn your labels into one-hot vectors
    train_labels = tf.one_hot(train_labels, 4)
    test_labels = tf.one_hot(test_labels, 4)
    return train_inputs, train_labels, test_inputs, test_labels

if __name__ == '__main__':
    # make a pickle file from the dataset
    data_folder = '../data'
    create_pickle(data_folder)
