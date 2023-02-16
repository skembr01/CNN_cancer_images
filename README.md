## Project

In this project from Kaggle, I am creating a convolutional neural network (CNN) to classify whether the objects in the scan are metastatic or not. These pictures are taken from pathology scans and the Kaggle dataset is a subset of the full data at https://github.com/basveeling/pcam. A tumor is classified as positive for cancer (metastatic) if there is a patch 32x32 pixel patch of the image that contains at least one pixel of tumor tissue.

There are 220,025 training images as well as corresponding data that labels each photo in a binary manner as being metastatic. Then, there are 57,458 images that are utilized as the test set for the project. Hence, I will train my CNN on the provided training data and then make a production for each class of the test data using my model. These predictions will then be uploaded to Kaggle and tested for their accuracy.

## Data

I imported the training dataframe from a csv and then converted the label column to a string type. Further, I added '.tif' to the end of each file id so that the images could be read in. Further, I shuffled the data in the training df, so that my model would not train on the ordering of the data, I also utilized a random state so that my data was reproducible. Further, I created a test dataframe of the test images. Lastly, I set a batch size of 64 that I will use in image generation and in modeling.

## EDA
Our first EDA comes from Kaggle as we are told there are no duplicate files in the data. Then, I wanted to start by checking for null values. To do this, I looked at df.info() and saw we had no non-null values. I further showed the df.describe() just to confirm there were no duplicates.

I then wanted to check to see if our data was unbalanced among the classes. Thus, I created a bar chart for the counts of each class and we got that there were approximately 57% of class 0 and 43% of class 1. This indicates a slight imbalance, but it is not enough that corrective action is needed. 

Then, I showed 9 random images from the training data (from the above shuffle) just to give an idea of what the photos represent. However, when I was implementing this, I found there was a file that contained an image which was nearly entirely black and clearly would not provide value to the task at hand. Hence, I removed this file from training and I also showed this image to explain why I removed it.

## Preparing Data to Model

I began by splitting the training into 80% training and 20% validation, so that the model would have something by which to validate its results.

Then, I utilized ImageDataGenerator from keras and rescaled the data by dividing each value on the RGB scale by 255 and then set the generator for a 80/20 training split. This function allowed me to grab the images that were found in the 'id' column in both the training and validation df's. In these generators, I also shuffled the data again, normalized by the standard deviation of the features, and set the target size from (96,96) to (64, 64) for optimal modeling. With our images in their corresponding training and validation sets, we are ready to model.


## Modeling

I utilized keras with a tensorflow backend. Further, I utilized tensorflow metal on my local machine, so that I could utilize its GPU and speed. To begin with, I had to set the layers for my sequential model. Based on the tip given in lecture, I utilized a pattern of Conv > Conv > MaxPool, which I repeated 3 times. Further, I added BatchNormalization and Dropout layers after my first two patterns. This allowed for better speed given that this model was complex for my local machine. After the 3rd iteration of my pattern, I then utilized a Flatten layer to place the data in a format that could be utilzied by a Dense layer. Finally, I utilized a Dense layer with a sigmoid activation function given that this was. binary classification. 

In terms of hyperparameters, each kernel size for the Conv layers was (3x3) and relu was used for each activation function. In the first iteration of Conv layers, I utilized 32 filters for each layer and then for the last two iterations I used 64 filters for each layer. This was done as we learned in class that the initial layers look at smaller points of the photo and the deeper you go the more a layer can determine. In terms of the MaxPooling layer I utilized pool sizes of (2,2) as recommended in lecture. 

With the layers of the model declared, the next step was to work on the model's compilation. To begin with, I set the optimizer to be Adam and utilized amsgrad to help with Adam convergence. In this optimizer, I set the initial learning rate to 0.0001 and further used a learning rate scheduler. This scheduler was made in such a way the first epoch utilized the initial learning rate and for each subsequent epoch, the learning rate decreased exponentially. Along with this optimizer, the loss used in the compilation was binary crossentropy, given the problem being of binary classification. Then, the metrics the model compiled on were the overall accuracy of classification as well as the area under ROC curve, to look at more than just pure accuracy.

Finally, I needed to fit the model to the training data. Here, the model utilized the training and validation image generators, then I set the batch size to be 64 and set the number of epochs to be 15. However, I also set an early stopping function with a patience of 2 epochs. Hence, if no improvement in metrics was made within 2 epochs, the model would stop. Here, I also made a callback to the learning rate scheduler. 

This model generated a public accuracy of 0.8735 and a private accuracy of 0.8245 on Kaggle and these were the best results I received after numerous hyperparameter tunings. 


## Prediction and Submission

Once the model had been trained, we used it to generate class predictions for the test images. I began to do this by using another image generator. Here, we again rescaled, we utilized an image size of (64,64), and we did not shuffle here as this would provide no advantage. Once the generator was complete, the model made its predictions. 

Then, we matched the prediction to the format of the sample submission by changing the id's and by marking a label as 1 if the prediction was >= 0.5 and 0 else. Finally, I connected to the Kaggle API and uploaded my data through this.



