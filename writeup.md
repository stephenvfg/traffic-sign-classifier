# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

* This document is the writeup.
* Link to all my source code is [here](https://github.com/stephenvfg/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

After unpacking the provided dataset I used pandas and other native functions to explore the data.
* There are **34,799 images** in the training set.
* There are **4,410 images** in the validation set.
* There are **12,630 images** in the test set.
* The images are **32 by 32 pixels** with three color channels (RGB)
* There are **43 different classes** or types of traffic signs found in the data sets.

#### 2. Include an exploratory visualization of the dataset.

I sampled one image from the dataset for each unique class. This provides a glimpse into the differences in each traffic sign image.

<<<<<PLACE FULL DATA SET HERE>>>>>>
  
To better understand what the data looks like I plotted a histogram to understand the frequency and distribution of each traffic sign type across all three of the datasets. What I'm seeing is that there are some classes that occur significantly less frequently than others in the data which may make it more challenging to confidently predict those classes once the model is trained. The distributions appear similar across the three datasets.

<<<<<<<PLACE DATA HISTORGRAM HERE>>>>>>>>
  
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

My image pre-processing pipeline was inspired by the research paper "Multi-column Deep Neural Networks for Image Classification" ([link](https://paperswithcode.com/paper/multi-column-deep-neural-networks-for-image)). The pipeline implemented the following steps for each image in the dataset:
1. Convert the image from RGB to Grayscale.
2. Increase the contrast of the image via the Imadjust function.
3. Further improve the contrast via Histogram Equalization.
4. Reduce information loss from step 3 by applying Contrast Limited Adaptive Histogram Equalization.
5. Apply Local Contrast Normalization.
6. Finally, center the image values around 0 between -1 and 1.

<<<<<<<PLACE PIPELINE HERE>>>>>>>>>>

#### OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.

I did not generate additional data to augment my training set. If I were to do so, I would have taken the following approaches:
* Augment the data for each class by taking a sample of existing data and applying a Gaussian blur to those images and using that to create new images of the same class.
* Similarly, taking a sample of existing data and applying slight rotations (10-30 degrees) and translations (few pixels in different directions) those images and using that to create new images of the same class.
* In cases where the training data is severely lacking in images of a certain class, I'd repeat the above efforts to even out the distribution.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model was largely based on the LeNet Convolutional Neural Network studied in the lessons. A few adjustments were made in an effort to improve the accuracy of the predictions. These departures from out-of-the-box LeNet are noted in the right table column.

| Layer         		    | Input Shape	      | Output Shape	    | Description & Notes                                              |
|:---------------------:|:-----------------:|:-----------------:|:----------------------------------------------------------------:| 
| Input         		    | N/A           		| 32 x 32 x 1			  | Convert from RGB to Grayscale in preprocessing                   |
| Convolutional #1   	  | 32 x 32 x 1	    	| 28 x 28 x 16		  | 5x5 kernel, 1x1 stride, valid padding. Increased depth 6 -> 16   |
| Activation            | 28 x 28 x 16      | 28 x 28 x 16      | ReLu activation                                                  |
| Pooling #1            | 28 x 28 x 16      | 14 x 14 x 16      | Max pooling with 2x2 kernel, 2x2 stride, valid padding           |
| Convolutional #2   	  | 14 x 14 x 16	   	| 10 x 10 x 32		  | 5x5 kernel, 1x1 stride, valid padding. Increased depth 16 -> 32  |
| Activation            | 10 x 10 x 32      | 10 x 10 x 32      | ReLu activation                                                  |
| Pooling #2            | 10 x 10 x 32      | 5 x 5 x 32        | Max pooling with 2x2 kernel, 2x2 stride, valid padding           |
| Flatten               | 5 x 5 x 32        | 800               | Scaled up from 400 -> 800 due to previous depth increases        |
| Fully connected #1    | 800               | 240               | Scaled up from 120 -> 240 due to previous depth increases        |
| Activation            | 240               | 240               | ReLu activation                                                  |
| Dropout               | 240               | 240               | Added to improve accuracy, reduce overfitting. Rate = 0.5        |
| Fully connected #2    | 240               | 168               | Scaled up from 84 -> 168 due to previous depth increases         |
| Activation            | 168               | 168               | ReLu activation                                                  |
| Dropout               | 168               | 168               | Added to improve accuracy, reduce overfitting. Rate = 0.5        |
| Fully connected #3    | 168               | 43                | Final fully connected layer to return the model logits           |

Several other adjustments were explored but ultimately dropped since they either hurt the model accuracy or added complexity while doing nothing to improve the output. For example:
* I explored using a Linear Scaled Hyperbolic Tangent as an activation function instead of ReLu ([inspired from here](https://forums.fast.ai/t/lisht-linear-scaled-hyperbolic-tangent-better-than-relu-testing-it-out/44002)).
* I added dropouts with low dropout rates after the hidden layers. This actually made my accuracy improvement rate less stable from epoch to epoch.
* Instead of Max pooling I tried to use Average pooling but there was no benefit.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train my model I used the following configuration:
* **I increased my Epochs from 10 to 32**. I saw a significant increase in accuracy and reduction in loss by doing this, although my model took much longer to train.
* For the same reason as above I **cut my batch size in half down to 64**.
* I started with a **learning rate of 0.001**. At the end of every epoch I multiplied the learning rate by a **decay parameter of 0.96**. This enabled me to add several more epochs and for the adjustments to my weights/biases to become more and more precise at each iteration.
* I used the out-of-the-box **Adam apative learning rate optimization algorithm**.

During training, for each epoch I would divide my training data into small batch sizes and run those batches thrrough my training model. I would train my model on the small batches until I iterated through the entire training dataset. At this point I would calculate the accuracy and loss values for that epoch, then slightly decay the learning rate, and then move on to the next epoch. At the end of the last epoch I would declare the model complete and store the calculated weights/biases.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I completed my model training with a **validation set accuracy of 96.3%** and a **test set validation of 94.2%**. You can see the improvement of my model's performance over time (with each epoch) in terms of validation set accuracy and loss below.

<<<<<<<< INCLUDE PEROFMRNACE PER EPOCH>>>>>>>>

To start building my model, I chose to begin with the LeNet model architecture we studied in class. Using the **out-of-the-box LeNet architecture with grayscale input images that were normalized** between (-1, 1) I was already able to reach a validation accuracy of **~90%**. Improving the accuracy from there was the big challenge of this project.

There were two primary approaches that I knew of to improve the accuracy. Approach #1 was to enhance the data **via image pre-processing**. Using the pipeline discussed above I was able to bring the validation accuracy **above 93%**.

**Some approaches did not work:**
* During my first attempts to improve the model, I swapped the ReLu activation functions with LiSHT activation functions. Doing this reduced my model's accuracy so I quickly scrapped this approach.
* I attempted to manipulate image data (via rotations and translations) in-between training epochs. I'm not sure if my implementation was off or if this is simply a bad idea but it tanked my validation accuracy below 80%.
* I added dropout after the max pooling functions thinking that it would improve accuracy. In fact, this was too soon in the model to add dropout (at a high rate of 50%) and it also crippled my model and reduced validation accuracy.

**Other approaches worked better to achieve a validation accuracy above 96%:**
* Increasing epochs and decreasing the batch sizes (to reasonable limits) brought up my model's accuracy.
* After a certain number of epochs, my validation accuracy would jump back and forth between the same range of accuracies because the learning rate was too high. I applied a learning rate decay of 0.96 per epoch so that the model adjustments would become more precise in the later iterations and my model would be able to continue improving.
* I did not change the order of type of layers in the LeNet model, but I did make the convolutional layers deeper (6->16 and 16->32). This improved the model accuracy but also may have increased risk of overfitting. I accounted for this in my next adjustment.
* Adding dropouts after the fully connected layers significantly improved accuracy and offset some of the risk of overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found several German traffic sign images online. I took five that I felt confident my model could predict and one that I thought it wouldn't be able to predict. I cropped the images down to 32 by 32 pixels so that I could use them in my model and I manually labelled them according to the classes my model knows.

<<<<<<< NEW IMAGES HERE >>>>>

The fourth image in the dataset will be difficult to classify for two primary reasons. First, the image has a secondary sign ("Schule") included within the first sign that may confuse my model. Second, the graphic art of the two people walking across the street looks slightly different compared to the images in the training data.

The five other images very closely resemble what we have in the data set and should be simple to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

| Image         	    	    | Prediction	            | 
|:--------------------------:|:-----------------------:| 
| Turn Left         		    | Turn Left           		| 
| No Passing         		    | No Passing          		| 
| Speed Limit 30         	 | Speed Limit 30          	| 
| Children Crossing          | Beware of Ice         | 
| Speed Limit 100          | Speed Limit 100          | 
| Road Work Ahead         	| Road Work Ahead        	| 

The model **accurately predicted 5 out of the 6** new images - and as I thought, it wrongly predicted the fourth image that looked a little different than what the model is used to seeing.

That being said, the model acheived an **83.33% accuracy rate on these images** compared to the 94.2% test set validation rate. This is slightly lower than expected but the sample size is so small that it doesn't feel surprising.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the signs that my model accurately identified, the softmax probability came extremely close to 100% for that classification. For the incorrect image classification the probability was barely over 50%. If I were to use this model in a production environment I could envision several rules and safeguards that would be useful where I could ignore classifications that had softmax probabilities below some threshold.

| Sign          | #1 Prediction & Prob. | #2 Prediction & Prob. | #3 Prediction & Prob. | #4 Prediction & Prob. | #5 Prediction & Prob. |
|:-------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| Turn Left         | 34 --> 99.99816%  | 11 --> 0.00117%       | 38 --> 0.00054%       | 40 --> 0.00009%       | 14 --> 0.00004%       |
| No Passing        | 9 --> 100.00000%  | 35 --> 0.00000%       | 10 --> 0.00000%       | 16 --> 0.00000%       | 12 --> 0.00000%       |
| Speed Limit 30    | 1 --> 99.99999%   | 4 --> 0.00001%        | 5 --> 0.00000%        | 0 --> 0.00000%        | 7 --> 0.00000%        |
| Children Crossing | 30 --> 56.31258%  | 12 --> 36.47871%      | 20 --> 6.64123%       | 24 --> 0.18723%       | 25 --> 0.12744%       |
| Speed Limit 100   | 7 --> 100.00000%  | 40 --> 0.00000%       | 5 --> 0.00000%        | 1 --> 0.00000%        | 8 --> 0.00000%        |
| Road Work Ahead   | 25 --> 100.00000% | 22 --> 0.00000%       | 29 --> 0.00000%       | 24 --> 0.00000%       | 18 --> 0.00000%       |

[Check out the sign names and code numbers list as reference.](https://github.com/stephenvfg/traffic-sign-classifier/blob/master/signnames.csv)
