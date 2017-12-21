# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./DataStats.png "Visualization"
[image2]: ./ColorAndGray.png "Grayscaling"
[image4]: ./WebImages/TestImage1.jpg "Traffic Sign 1"
[image5]: ./WebImages/TestImage2.jpg "Traffic Sign 2"
[image6]: ./WebImages/TestImage3.jpg "Traffic Sign 3"
[image7]: ./WebImages/TestImage4.jpg "Traffic Sign 4"
[image8]: ./WebImages/TestImage5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shebl99/CarND-Traffic-Sign-Classifier-Project.git)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the normal python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows the distribution of the data on all the classes. It is clear that the training, test, and validation sets generally share the same distribution of examples among the classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this makes the data easier for the network to learn since it has to care about the shape of the sign not its color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because numerically it is better to work with numbers that are on the same order of magnitude and roughly evenly distributed around zero. This makes the optimizer converge faster.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet architecture with a small modification. I added a couple of dropout layers.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| input 400, ouput 120 							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 120, ouput 84 							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 84, ouput 43 							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:
learning rate: 0.002
epochs: 600
Keep probability: 0.4
Optimizer: AdamOptimizer
I didn't need to use batches. My machine was able to handle all the data at once. However, I included the code required to do the batches but commented it.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.38%
* validation set accuracy of 93.78%
* test set accuracy of 92.59%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I started with the LeNet architecture as it was the one suggested in the course.
* What were some problems with the initial architecture?
It couldn't reach the required accuracy
* How was the architecture adjusted and why was it adjusted? 
I added two dropout layers in the architecture. This is to have some redundancy in the network to reach higher accuracy and avoid overfitting.
* Which parameters were tuned? How were they adjusted and why?
The number of epochs was increased until the results couldn't enhance anymore. This is to avoid overfitting.

The accuracy results were close in the validation and the test sets. They were also close on the web images so I believe the model is working correctly.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I don't think there is any difficulty for the model to detect the images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead 		| Turn right ahead								| 
| Speed limit (60km/hr)	| Speed limit (50km/hr)							|
| Pedestrians			| Pedestrians									|
| Children crossing		| Children crossing				 				|
| Road work 			| Road work         							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.59%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The model was always certain about the probability it provides. In all the images, the predicted class had a probability of 1.0 and all other classes had a probability of 0.0

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


