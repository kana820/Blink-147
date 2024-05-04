import tensorflow as tf

def get_labels(true_path, btl_path):
    '''Retrieves labels from true and predicted.

    :param true_path: path to true images and labels, btl_path: path to pred images and labels
    :return: tuple containing list of true labels and list of pred labels
    '''
    with open(true_path) as file:
        examples = file.read().splitlines()

    with open(btl_path) as file:
        examples2 = file.read().splitlines()
    
    # convert labels in string to ints for true labels
    image_names_to_labels = {}
    for example in examples:
        img_name, label = example.split(',')
        if label == "left":
            image_names_to_labels[img_name] = 1
        elif label == "right":
            image_names_to_labels[img_name] = 2
        elif label == "up":
            image_names_to_labels[img_name] = 3
        elif label == "blink":
            image_names_to_labels[img_name] = 4

    # convert labels in string to ints for pred labels 
    image_names_to_labels2 = {}
    for example in examples2:
        img_name, label = example.split(',')
        if label == "left":
            image_names_to_labels2[img_name] = 1
        elif label == "right":
            image_names_to_labels2[img_name] = 2
        elif label == "up":
            image_names_to_labels2[img_name] = 3
        elif label == "blink":
            image_names_to_labels2[img_name] = 4
        # included "none" label if BTL could not detect a face
        elif label == "none":
            image_names_to_labels2[img_name] = 5

    true_list = []
    btl_list = []

    # checks if images in BTL images are in true images and 
    # appends labels to true and pred lists in order
    for img_name in image_names_to_labels2.keys():
        if img_name in image_names_to_labels.keys():
            true_list.append(image_names_to_labels[img_name])
            btl_list.append(image_names_to_labels2[img_name])

    return true_list, btl_list

true_labels, btl_labels = get_labels("../data/test_labels.txt","computer_vision/btl_labels_test.txt")

def evaluate(true_labels, pred_labels, batch_size = 16):
    '''Evaluates MSE loss and accuracy of Blink-To-Live model.

    :param true_labels: true labels, pred_labels: predicted labels
    :return: tuple of mean loss and mean accuracy
    '''
    combined_data_set = tf.data.Dataset.from_tensor_slices((true_labels, pred_labels))
    batches = combined_data_set.batch(batch_size)

    losses = []
    accuracies = []

    for true_batch, pred_batch in batches:
        loss = tf.keras.losses.mean_squared_error(true_batch, pred_batch)
        losses.append(tf.cast(loss,tf.float32))

        accuracy = tf.reduce_mean(tf.cast(tf.equal(true_batch, tf.round(pred_batch)), tf.float32))
        accuracies.append(tf.cast(accuracy, tf.float32))
        print("Loss:" + str(tf.cast(loss,tf.float32)), "Accuracy:" + str(tf.cast(accuracy, tf.float32)))

    mean_loss = tf.cast(tf.reduce_mean(losses), tf.float32)
    mean_accuracy = tf.cast(tf.reduce_mean(accuracies), tf.float32)

    print("Loss:" + str(mean_loss), "Accuracy:" + str(mean_accuracy))
    return mean_loss, mean_accuracy

evaluate(true_labels, btl_labels)