# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

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
| Fully connected       | 800               | 240               | Scaled up from 120 -> 240 due to previous depth increases        |
| Activation            | 240               | 240               | ReLu activation                                                  |
| Dropout               | 240               | 240               | Added to improve accuracy, reduce overfitting. Rate = 0.5        |
| Fully connected       | 240               | 168               | Scaled up from 84 -> 168 due to previous depth increases         |
| Activation            | 168               | 168               | ReLu activation                                                  |
| Dropout               | 168               | 168               | Added to improve accuracy, reduce overfitting. Rate = 0.5        |
| Fully connected       | 168               | 43                | Final fully connected layer to return the model logits           |

Several other adjustments were explored but ultimately dropped since they either hurt the model accuracy or added complexity while doing nothing to improve the output. For example:
* I explored using a Linear Scaled Hyperbolic Tangent as an activation function instead of ReLu ([inspired from here](https://forums.fast.ai/t/lisht-linear-scaled-hyperbolic-tangent-better-than-relu-testing-it-out/44002)).
* I added dropouts with low dropout rates after the hidden layers. This actually made my accuracy improvement rate less stable from epoch to epoch.
* Instead of Max pooling I tried to use Average pooling but there was no benefit.
