# GTSRB---German-Traffic-Sign-Recognition-Benchmark-classification-project

Context

The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. We cordially invite researchers from relevant fields to participate: The competition is designed to allow for participation without special domain knowledge. Our benchmark has the following properties:

    Single-image, multi-class classification problem
    More than 40 classes
    More than 50,000 images in total
    Large, lifelike database
the link of the datasets 'https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download'
the steps in the project:
   1- Import Libraries: Import the necessary libraries for image processing, data manipulation, and building the CNN model.
   
   2- Data Preprocessing:
        Define the Path: Set the path to your dataset on Google Drive.
        Read Images: Loop through each class directory, read, resize, and convert images to NumPy arrays.
        Normalize Image Data: Normalize pixel values to the range [0, 1].
        One-Hot Encode Labels: Convert labels to one-hot encoding.
        Train-Validation Split: Split the data into training and validation sets. 
        
  3- Define the CNN Model:
        Convolutional Layers: Use Conv2D layers with ReLU activation followed by MaxPooling2D layers.
        Flatten Layer: Use Flatten to convert 2D feature maps to 1D feature vectors.
        Fully Connected Layers: Use Dense layers with ReLU activation and Dropout for regularization.
        Output Layer: Use a Dense layer with softmax activation for multi-class classification.
        
  4-  Compile the Model: Compile the model with the Adam optimizer, categorical crossentropy loss, and accuracy metric.
  
  5-  Train the Model: Train the model with the training data, validating on the validation data.
  
  6- Plot Training History: Plot the training and validation accuracy and loss over epochs.
  
  7-  Evaluate the Model: Evaluate the trained model on the validation set and print the results.

This code will read the GTSRB dataset, preprocess the images, build and train a CNN model, and visualize the training process. The final evaluation step provides the validation loss and accuracy.
