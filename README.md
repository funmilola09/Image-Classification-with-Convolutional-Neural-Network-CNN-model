# Image-Classification-with-Convolutional-Neural-Network-CNN-model
Build, compile and train a Convolutional Neural Network (CNN) model. This model will classify images in the CIFAR10 dataset into their distinct categories.





The steps that are involved in building and training this model are as follows:
1.Import necessary libraries, and load the dataset. This dataset should consist of labelled samples for both training and testing.
2.Preprocess the data. This involves normalising the pixel values, along with transformations for data augmentation such as rotating, flipping, cropping etc.
3.Build the model (Constructor Stage). In Keras, a Sequential model is a linear stack of layers, and you can add layers to it using the add() method. The first layer in the model should specify the input shape. For a CNN model, we start by creating the convolutional and pooling layers, after which we use a flatten layer followed by the dense, fully connected layers.
4.Compile the model (Compilation Stage). This step involves specifying the loss function, the optimizer, and any metrics that you want to track during training.
5.Train the model (Training Stage). Use the fit() method to train the model on the training data. You can also specify the validation data here. This may also involve hyperparameter tuning, if required.
6.Evaluate the model (Evaluation Stage). Use the evaluate() method to evaluate the model on the test data. This will give you the accuracy and other metrics that you specified earlier.
7.Make predictions. Use the predict() method to make predictions on new data.
8.Visualise the results. Use tools such as the Confusion matrix, classification report, Matplotlib and Seaborn to visualise the results, for example, plot the training and validation loss, plot the confusion matrix, etc.



The Dataset
The Canadian Institute for Advanced Research (CIFAR10) image dataset was used. This consists of 60,000 32X32 colour images having 10 classes with 6000 images per class. The dataset has been split into a training set containing 50,000 images and a testing set containing 10,000 images.
Dataset Source: https://www.cs.toronto.edu/~kriz/cifar.html
Dataset Description
Number of classes: 10
Total number of images: 60000 32x32 colour images
Number of images per class: 6000 images per class
Number of training images: 50000
Number of test images: 10000
