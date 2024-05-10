# Blink-147
## Deep Learning Model to Classify Eye Movements for Eye-Based Communication with Total Paralysis Patients
![Blink-147 Poster](https://github.com/kana820/Blink-147/assets/107338457/baddc37e-93e9-45bd-b081-62e0b42ced5f)

# Dataset
__Closed Eyes In The Wild__    
__\[Citation]__ F.Song, X.Tan, X.Liu and S.Chen, Eyes Closeness Detection from Still Images with Multi-scale Histograms of Principal Oriented Gradients, Pattern Recognition, 2014.
[Access from here](http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html)

# How to Run
We developed three techniques to prevent our CNN model from predicting only the dominant class (blink) due to the data imbalance: Weighted Loss, Weighted Sampling, and Data Augmentation. There will be a few differences in the ways to run the model, depending on which technique you would like to utilize.  
1. Install the dependencies    
2. Preprocess Data: download the dataset from the above link and combine images in the ClosedFace and OpenFace folders. All the labels for relevant pictures (either left, right, up, or blink) are stored in 'train_labes.txt' in the data folder. Create a pickle file by doing one of the following.    
   (1) if you are using the original dataset, run 'preprocess.py' after uncommenting the code in the main and changing the path to the dataset if necessary.
   (2) if you wish to have the augmented dataset, run 'preprocess_augmented.py' after uncommenting the code in the main and changing the path to the dataset if necessary. It will upsample images in the three underrepresented classes (left, right, and up) by flipping, cropping, etc. and downsample images from the blink class.   
   Depending on which preprocess file you use, change the import statement for the preprocess file at the top of 'main.py' and the name of the data pickle file in the main function if necessary.   
4. Choose the architecture: Our model is originally attention-based. If you want to use our model without those attention layers, change the import statement for the model file to 'model_without_attention'. The model in 'model_without_attention.py' has more dropout layers and regularizaions on the last three convolutional and the first dense layers.   
5. Choose the loss function: The loss function is defined in the Model class in 'model.py'. There are two sets of code, one for the standard cross entropy loss and the other for the weighted cross entropy loss. Uncomment and use one of these. You may change the class weights, according to the proportion of each class in your dataset.
6. Weighted Sampling: If you wish to use the technique of wieghted sampling, uncomment line 75 in 'main.py' to call the function for training.
7. Visualization: we prepared visualization functions for several features of the model: visualize_cnn_layer (output from the second convolution layer), visualize_attention (attention maps), visualize_train_test_acc (accuracy on training vs testing over epochs), visualize_loss (loss over epochs), and visualize_acc (accuracy over epochs). Comment out/uncomment the lines calling the functions in 'main.py' if applicable.
9. Run 'main.py'
