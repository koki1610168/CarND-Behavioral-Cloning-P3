# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./model_summary.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model is Tensorflow Background of the Nvidia designed nerual network architecture. It analyzes where the car is, and where to go next.

#### 2. Attempts to reduce overfitting in the model

I split the data into training samples and validation samples. I used generator-yeil syntax in order to reduce the computation time.

Moreover, I used Dropout of 0.5. When I used 0.2 Dropout, I tends to overfit the data, so I prefer to used 0.5 one.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I attempted to collect my own data and try my model on my computer, though my gpu is not very suitable for this heavy calculation, so I used the workspace's data to train.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Although I have employed some other network architecture, the Nvidia one works the best in terms of the loss.

At the end of the process, the car autonomously runs around the course without diving into the lake.

#### 2. Final Model Architecture

1.Convolution -  filters=24, kernel_size=5, strides=2, activation='relu'
2.Convolution -  filters=36, kernel_size=5, strides=2, activation='relu'
3.Convolution -  filters=48, kernel_size=5, strides=2, activation='relu'
4.Convolution -  filters=64, kernel_size=3, strides=1, activation='relu'
5.Convolution -  filters=64, kernel_size=3, strides=1, activation='relu'

Flatten 

followed by Dense layers

![alt text][image1]
#### 3. Creation of the Training Set & Training Process

When I was recording the driving, I paid attention on goin middle all the time. After that process, I recorded recovery process in case the car gets off the road.
