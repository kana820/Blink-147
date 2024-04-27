import numpy as np
import pickle
import tensorflow as tf
import os
from PIL import Image

def load_data(data_folder):
    '''
    Method that was used to preprocess the data in the data.p file. You do not need
    to use this method, nor is this used anywhere in the assignment. This is the method
    that the TAs used to pre-process the Flickr 8k dataset and create the data.p file
    that is in your assignment folder.


    Feel free to ignore this, but please read over this if you want a little more clairity
    on how the images and captions were pre-processed
    '''


    test_file_path = f'{data_folder}/test_labels.txt'
    train_file_path = f'{data_folder}/train_labels.txt'


    with open(test_file_path) as file:
        test_examples = file.read().splitlines()
       
    with open(train_file_path) as file:
        train_examples = file.read().splitlines()
   
    # map each image name to its captions
    test_image_names_to_labels = {}
    for test_example in test_examples:
        test_img_name, test_label = test_example.split(',')
        test_image_names_to_labels[test_img_name] = test_image_names_to_labels.get(test_img_name, []) + [test_label]
       
    train_image_names_to_labels = {}
    for train_example in train_examples:
        train_img_name, train_label = train_example.split(',')
        train_image_names_to_labels[train_img_name] = train_image_names_to_labels.get(train_img_name, []) + [train_label]


    test_img_fle_path = f'{data_folder}/test'
    train_img_file_path = f'{data_folder}/train'


    test_images = get_image_from_dir(test_img_fle_path, test_image_names_to_labels.keys())
    train_images = get_image_from_dir(train_img_file_path, train_image_names_to_labels.keys())


    # returns all captions for all images
    def get_all_labels(image_names, image_names_to_labels):
        to_return = []
        for image in image_names:
            label = image_names_to_labels[image]
            to_return.append(label)
        return to_return
   
    test_labels = get_all_labels(test_image_names_to_labels.keys(), test_image_names_to_labels)
    train_labels = get_all_labels(train_image_names_to_labels.keys(), train_image_names_to_labels)

    return dict(
        test_labels          = test_labels,
        test_images            = test_images,
        train_labels          = train_labels,
        train_images            = train_images,
    )

def get_image_from_dir(data_folder, image_names):
    images = []
    for image in image_names:
        image_path = os.path.join(data_folder, image)
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                old_size = img.size
                ratio = 100 / max(old_size)
                new_size = tuple([int(x*ratio) for x in old_size])
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                new_img = Image.new("RGB", (100, 100))
                new_img.paste(img, ((100 - new_size[0])//2,
                    (100 - new_size[1])//2))
                img_array = np.array(new_img)
                images.append(img_array)
        else:
            print(image)
    return images

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
    train_inputs = unpickled_file.get('train_images')
    train_labels = unpickled_file.get('train_labels')
    test_inputs = unpickled_file.get('test_images')
    test_labels = unpickled_file.get('test_labels')
    
    def reshape_images(images):
        images = np.array(images)
        images = images / 255
        images = tf.reshape(images, (-1, 3, 100, 100))
        images = tf.transpose(images, perm=[0,2,3,1])
        # Now in the shape (num_examples, 100, 100, 3)
        return images
    
    train_inputs = reshape_images(train_inputs)
    test_inputs = reshape_images(test_inputs)
    print("num of train samples", len(train_inputs))
    print("num of test sample", len(test_inputs))

    # Convert labels in string to ints
    # Left=0, Right=1, Up=2, Blink=3
    label_map = {"left": 0, "right": 1, "up": 2, "blink": 3}
    train_labels = [label_map[label[0]] for label in train_labels]
    test_labels = [label_map[label[0]] for label in test_labels]
    # Turn your labels into one-hot vectors
    train_labels = tf.one_hot(train_labels, 4)
    test_labels = tf.one_hot(test_labels, 4)
    return train_inputs, train_labels, test_inputs, test_labels

if __name__ == '__main__':
    # make a pickle file from the dataset
    # data_folder = 'data'
    # create_pickle(data_folder)
    pass