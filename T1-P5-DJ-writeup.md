**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_non_car.png
[image21]: ./output_images/hog0.png
[image22]: ./output_images/hog1.png
[image23]: ./output_images/hog2.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/sliding_limit.png
[image51]: ./output_images/pipeline1.png
[image52]: ./output_images/pipeline2.png
[image53]: ./output_images/pipeline3.png
[image54]: ./output_images/pipeline4.png
[image55]: ./output_images/pipeline5.png
[image56]: ./output_images/pipeline6.png
[image57]: ./output_images/pipeline7.png
[image61]: ./output_images/falseposhls.png
[image62]: ./output_images/falseposluv.png
[image7]: ./output_images/trainpipeline.jpg

[video1]: ./project_output_ycrcb_th4.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

The write below describes how I approached  the project. I have used functions that were described in the lectures as well as by Ryan in his Q&A video session.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features` defined in **HOG Feature Extraction Function** section of the `T1-P5-DJ.ipynb`.

The first step towards experimentation required reading sample `vehicle` and `non-vehicle` image.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![sample car and non-car images][image1]

I then explored different color spaces (e.g. HLS, LUV, YUV etc) and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I extracted hog features on above images and plotted them to get a feel for what the `skimage.hog()` output looks like.

Below is the example using the `YCrCb` color space, nipy-spectral cmap and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for all channels of image. 
Note: I am using NIPY-Spectral cmap to allow for easier visualization by the human eye.

![hog features][image21]

![hog features][image22]

![hog features][image23]

#### 2. Explain how you settled on your final choice of HOG parameters.

One of the over-arching focus I had was to maintain balance between accuracy on the test set, and the size of the feature set. This is critical since training time is directly related to the size of the feature set. 

The first step here was for me to confirm what is indicated in the HOG paper regarding orientation. I played with 3 values of orientation and identified that orient=9 gives the best accuracy which maintaining reasonable size of the feature set. Here are the results captured from the three tuning measurements:-

`  Using: 12 orientations, 8 pixels_per_cell, 2 cells per block, 32 histogram bins, and (32, 32) spatial sampling
Feature vector shape: (11700, 10224)
My SVC score on test images:  0.988717948718`

`Using: 9 orientations, 8 pixels_per_cell, 2 cells per block, 32 histogram bins, and (32, 32) spatial sampling
Feature vector shape: (11700, 8460)
My SVC score on test images:  0.990427350427`

`Using: 6 orientations, 8 pixels_per_cell, 2 cells per block, 32 histogram bins, and (32, 32) spatial sampling
Feature vector shape: (11700, 6696)
My SVC score on test images:  0.986666666667`

As one can note my feature space reduced by 20% going from orientation of 12 to 9, while my accuracy improved. Further reduction of orientation deteriorated the accuracy a bit although it still reduced feature space substantially. I decided to use orientation of 9 as a best choice based on this data.

Next I played with  various colorspaces, and found false positives in using some of them. This is shown below:-

![False Positive HLS][image61]

![False Positive LUV][image62]

A lower count of pixels_per_cell increased run time and memory resources for hog extraction. I also found that using 3 color channels gave better accuracy than single channel hog features.  

Finally I settled on following hog features:-
* orientations    = 9
* pixels_per_cell = (8, 8)
* cells_per_block = (2, 2)
* color space = "YCrCb"
* channel used = "All" 

There are all my experimental subclips in the folder where I have experimented extensively with following additional parameters:-

 1. Thresholding 
 2. Scaling of window sizes
 3. Selecting the Q depth for frame to frame averaging

Here's a [link to my video using HLS mapping with lot of false positives. ](./project_output_hls.mp4)

Here's a [link to my video using YUV mapping with most false positives eliminated. ](./project_output_hls.mp4) 

However, I see that at between 26s-28s the classifier does not correctly identify the white car in the frame. Hence I continued to play with threshold and colorspaces to fix this one nagging issue. 

I have kept all my experimental subclips under the 'subclips' folder.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The feature extraction function `extract_features` uses multiple functions such spatial down sampling, color histogram and HOG to generate a feature vector for each of the images. I normalize the vectors before training them. The full pipeline is shown below.

![Training Pipeline][image7]

Training classifer code can be found at `TRAINING CLASSIFIER` section of `T1-P5-DJ.ipynb`. 
For classification, I used Linear SVM (Support Vector Machine) classifier with default options. I used following features for classifier following the guidelines. I made sure to normalize the parameters after getting all feature vectors ordered properly. 

1) Spatial Binning  

Here I resize the image to spatial size (32, 32) and then use its flattened array. The primary purpose of this step is to reduce the feature space to ensure reasonable training time.

2) Color Histogram  

Create a histogram of intensity in each channel with 32 bins and concatenate all these channels.

3) HOG  

I create hog of each channel as discussed in hog section of this document.

I standardize the data by using `StandardScaler` module of `sklearn.preprocessing`.   
I randomly split training and testing data (20%) for cross-validation. After training SVM on train data, we find accuracy of more than 98% on testing data. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I slide windows of size 64 * 64 across the image. In each portion of sliding window, I extract features and check for cars using trained model. I collect all car predicting boxes. I maintain an overlap of 50% between different windows which was able to give car detections with good runtime performance.   

To find cars with varying view sizes, I resize the image with different scales 1.1,1.5,1.8,2.0,2.5 and then run slide windows. I did not want to use very small window sizes since expectations for car to fit certain size. I also limited my search to be half way into image size to minimize false positives like finding cars where they are not expected to be (like on top of a try or in cloud formation in the sky).   

Implementation of sliding windows can be found at `find_cars` function of `T1-P5-DJ.ipynb`.  
Below is sample image of boxes found through sliding windows. As you can see, the cars in the opposite lanes are also identified by the classifier. 

![sliding windows][image3]

Below shows the range of the bounded search area of sliding windows in between road lanes used to improved performance of my pipeline.  Perhaps one can consider bounding this further but then position of the car in the lane can change from left most to the right most. This probably means only bound on the y-axis is meaningful. 

![sliding limit][image4]


#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

I create HOG of entire image once and just take its subsample on each sliding window to improve runtime performance. 

To overcome false detection, I limited search area of sliding windows in between road lanes as shown in below image as described in the previous section.

Also I create a heatmap as per count of boxes in each pixel and apply a threshold limit so that false detection is filtered out. Output of my pipeline for different test images is depicted below.  

![pipeline][image51]
![pipeline][image52]
![pipeline][image53]
![pipeline][image54]
![pipeline][image55]
![pipeline][image56]

See also my description in the section above **Section 2** which talks about how I went about improving performance of my classifier based on changing orientation and color mapping. I did not play with histogram parameters per Ryan's Q&A session and kept my focus on areas of maximum impact.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)    
Here's a [link to my final video result](./project_output_ycrcb_th4.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Check pipeline output image in last section.

I also keep a dequeue for last 10 frames and take their sum with a threshold to create smooth bouding box and avoid jitters in video. It also avoid glitches in car detection. Here is the code:-

`from scipy.ndimage.measurements import label
def get_labels(img, Q = None):
    bbox = find_bbox(img)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, bbox)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold-1)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)`
    
    if Q != None:
        if len(Q) == 10:
            Q.popleft()
        Q.append(heatmap)
        sum_heat = sum(Q)
        heatmap = apply_threshold(sum_heat, threshold+2)
    labels = label(heatmap)
    return labels

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Pipeline mentioned in this document is able to detect cars with great accuracy in most of frames of the submitted video. 

There are still some issues as described below:-

1) Cars are detected in the opposite direction   
2) Time lag between when car is able to get detected after entry into frame.    
3) separate bounding box of overlapping cars
4) False Negatives

![False Negatives][image57]


To make pipeline more robust, we can try following actions
1) Use richer vehicls/non-vehicles dataset. In my opinion this will have the most impact for better classification.
2) Use deep learning image classifier like AlexNet as an alternative 
2) Vary thresholds, hyperparameters and use better window averaging techniques for frame to frame matching. 
4) Experiment with different videos and road conditions
