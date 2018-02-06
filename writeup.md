# **Vehicle Detection Project**

The goal of this project is to detect vehicles in a video using the following steps:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Normalize the features and randomize a selection for training and testing and train a Linear SVM classifier.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png "Car"
[image2]: ./output_images/not_car.png "Not Car"
[image3]: ./output_images/car_hog.png "Hog Visualization of Car"
[image4]: ./output_images/not_car_hog.png "Hog Visualization of Not Car"
[image5]: ./output_images/slide_win_1.png "Sliding Windows 64x64"
[image6]: ./output_images/slide_win_2.png "Sliding Windows 96x96"
[image7]: ./output_images/slide_win_3.png "Sliding Windows 128x128"
[image8]: ./output_images/slide_win_4.png "Sliding Windows 224x224"
[image9]: ./output_images/slide_win_all.png "Sliding Windows Combined"
[image10]: ./output_images/bbox_1.png "Bounding Box Example"
[image11]: ./output_images/bbox_2.png "Bounding Box Example"
[image12]: ./output_images/bbox_3.png "Bounding Box Example"
[image13]: ./output_images/bbox_4.png "Bounding Box Example"
[image14]: ./output_images/bbox_5.png "Bounding Box Example"
[image15]: ./output_images/bbox_6.png "Bounding Box Example"
[image16]: ./output_images/heat1.png "Bounding Box and Heat Example"
[image17]: ./output_images/heat2.png "Bounding Box and Heat Example"
[image18]: ./output_images/heat3.png "Bounding Box and Heat Example"
[image19]: ./output_images/heat4.png "Bounding Box and Heat Example"
[image20]: ./output_images/heat5.png "Bounding Box and Heat Example"
[image21]: ./output_images/heat6.png "Bounding Box and Heat Example"
[image22]: ./output_images/label_img.png
[image23]: ./output_images/output.png
[video1]: ./videos/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

This is the writeup

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code to read the dataset is contained in lines 14 through 58 and 89 to 100 of `train.py`, and the code to extract HOG features is contained in lines 26 through 45 of `features.py`

I started by reading in the dataset of `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Fig1: Example Car Image in the dataset

![alt text][image2]

Fig2: Example Non Car Image in the dataset

I then explored different `skimage.hog()` parameters like `orientations`, `pixels_per_cell`, `cells_per_block` and also the optional parameters `block_norm` and `transform_sqrt`. For my experiments with hog features, I enabled the `transform_sqrt` as it applies power law (gamma correction) compression to normalize the image before processing, this reduces the effects of shadowing and illumination variations. The compression makes the dark regions lighter.

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations = 8`, `pixels_per_cell = (8, 8)` and `cells_per_block = (2, 2)`:

![alt text][image3]

Fig3: Hog features of a car image

![alt text][image4]

Fig4: Hog features of a not car image

#### 2. Explain how you settled on your final choice of HOG parameters.

I chose the hog parameters to balance speed and accuracy. The orientation value controls the granularity of binning gradient directions. I tried 8, 9 and 10 orientation bins, and got higher test set accuracy with higher orientation values, however I didn't notice any improvement in detection accuracy on test and project videos. I chose 8 orientation bins as it divides 360 degrees into 8 equal bins of 45 degrees, which I felt is sufficient to detect vehicles which mostly have square or a quadrilateral shape, furthermore, it also gives smaller hog features, so faster detection. I tried 8 and 16 pixels per cell and notices better training set accuracy and improved detection accuracy in test videos, so I decided to choose 16 pixels per cell. The orientation bins in a cell are normalized with the other cells in a block, I tried 1, 2 and 4 cells per block and got best results with 2 cells per block.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I started with extracting spatially binned, color histogram and hog features, and got high test set accuracy (around 99%), however, I noticed that the detection accuracy in the test videos was poor with lot of false positives. Moreover, with bigger features, the vehicle detector was very slow to process frames in test videos. So, I decided to use only hog features as it was fast and gave better detection accuracy in test videos. Since there are only two classes, car and not car, I trained the classifier using LinearSVC from scikitlearn. I also normalized the features

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I noticed that the cars only occur in the bottom part of the video frames, and the cars appear smaller as they move farther from us and closer to the horizon and bigger when they are closer to us. So, I scanned the image with smaller sliding windows of size 64x64 pixels near the horizon, 224x224 pixels near the camera and 96x96 pixels and 128x128 pixels in between. A sample scan region image can be seen below.

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

Fig5: (Clockwise) Sliding Windows 64x64, 96x96, 128x128, 224x224

![alt text][image9]

Fig6: All the sliding windows combined

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I started with extracting spatially binned, color histogram and hog features, and got high test set accuracy (around 99%), however, I noticed that the detection accuracy in the test videos was poor with lot of false positives. Moreover, with bigger features, the vehicle detector was very slow to process frames in test videos. So, I decided to use only hog features as it was fast and gave better detection accuracy in test videos. I also experimented with HSV, HLS, YCrCb and YUV color spaces and found best performance with HSV color space. Here are some example images:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](videos/output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I maintained a history of positive detections for previous 10 frames of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions which have been consistently found in last 10 frames.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image22]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image23]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems I faced in the project are mostly with the detection accuracy and removing false positives. Although I got high accuracy (around 99%) on test set by concatenating spatially binned features, color histogram features and hog features, it increased the false positives and drastically reduced the speed of detector. So I had to experiment with different parameters for hog features and spatially binned and color histogram features. After several experiments that I was able to achieve better and faster performance with just hog features.

The other problem I faced in the project is with filtering out the false positives and stability of bounding box around the detected vehicles. I noticed that even though the vehicles in video were detected, the bounding box changed size rapidly and wouldn't bound only some part of the car. To deal with this problem and to filter the false positives, I maintained a history of detected windows in last 10 frames of the video. Instead of generating heat map of the window detected in the current frame, I generated heat map of windows detected over the last 10 frames. The bounding boxes generated using this approach were more stable and accurate than before, furthermore it also succeeded in thwarting out the false positives.  

My pipeline may fail in situations where the vehicles are moving very fast relative to the camera such that the number of detected windows may fall below the detection threshold and are failed to detect. I also noticed that the dataset consisted mostly of images of back of the car, so I am not sure whether this program will be able to detect oncoming traffic. In addition to fast moving vehicles, shadows and illumination variations can cause issues with detecting vehicles. To deal with this problem I have enabled `transform_sqrt` in `skimage.hog()`, however, it may be hard to detect vehicles in dark, unless the dataset consists of images images of vehicles in badly lit scenes.

Given plenty of time to pursue, I would implement the below listed improvements
1. Use deep-learning based approach like R-CNN, YOLO, SSD to detect vehicles with more accuracy and in real-time.
2. Improve the false positive filtering method by weighing the positive detections based on time.
3. Implement vehicle tracking mechanism using Kalman Filter or some other method.
