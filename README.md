# Vehicle Detection Project

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[original]: ./test_images/test1.jpg
[YCr]: ./output_images/YCr.png
[hog]: ./output_images/hog.png
[ValidationCurve]: ./output_images/Validation_Curve.png
[LearningCurve]: ./output_images/Learning_Curve.png
[SlidingWindow]: ./output_images/sliding_widow.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### README 

### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images.

The code for this step is contained in the code cells 1-3 of the IPython notebook (*'Vehicle_Detection.ipynb'*).

Histogram of Oriented Gradients (HOG) feature extraction consists of two steps: 
  1. Data preprocessing - in order to train SVM classifier I used provided dataset of car and non-car images. The dataset has a relatively small size and can be efficiently processed in RAM. Therefore, I did not implement any methods (generators) for on the fly data processing and rather focused on sequential loading the dataset and preprocessing. The datset consists of 17,770 samples (208Mb). The second code cell describes the procedure of loading the dataset into memory.
  2. I then explored different color spaces. YCrCb space seemed to provide a good separation for various objects in color space. The chart below provides representations of the original image and its representation in Y, C, R challens:

![alt_text][original]

![alt_text][YCr]

As one can observe from the chart above, balck pixels related to black vehicle and white pixels related to white vehicle are clustered on the opposite sides of the chart, providing a good separation for the objects.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog]

#### 2. Explain how you settled on the final choice of HOG parameters.

In order to choose a good combination of parameters, I trained SVC classifier and measured its performance with a given HOG settings. As a result of experiments, the best choice of the features was:
* Color histograms
* Histogram of Oriented Gradients (HOG)
for YCrCb channels with the following parameters:
feature_config = {"orientations": 9, "pix_per_cell": 8,
                  "cell_per_block": 2, "spatial_size": (32, 32),
                  "hist_bins": 32, "cmap":"YCr"}

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In order to tune classifier it is important to answer 2 questions.
A) What are the best settings for the parameters of the classifier
B) Do we have enough data to train classifier well
For the purpose of this assignment we test LinearSVC classifier. This classifier depends only on penalty parameter C. In order to find the best settings for C parameter, I compared training and validation scores for various values of C (plot validation curve). To optimize performance of this process, I will limit the sample size to 1100 sampels.

The results of validation curve are presented below:
![alt text][ValidationCurve]

Based on the results of the validation curve analysis presented below, I conclude that LinearSVC classifier is not sensetive to the value of C parameter. Therefore, I suggest to use default C value = 1.

As the next step, it is important to check if we have valid amount of data to train our classifier. For this purpose I will analyze the performance of training and validation scores on various dataset sizes (Learning Curve analysis).

![alt text][LearningCurve]

Based on the results of Learning curve analysis, I conclude that our dataset is sufficient to train LinearSVC classifier. Based on the plot presented above, one can observe that the improvement of validation score plateaued at 10,000 samples (as oppose to 17,770 samples available in our dataset).

The analysis above provided in code cells 6-9 (*'Vehicle_Detection.ipynb'*).

The last step is to train our classifier and to analyze it's performance. For the analysis of classifier's performance I will use 3 metrics:
* accuracy - sum of true positives and negatives over the total number of elements.
* recall - Percentage of ground truth examples identified correctly
* precision - share of predictions was correct

Linear SVM demostrates the following results:
* Classifier accuracy - 99.16%
* Classifier recall - 99.08%
* Classifier precision - 99.19%

Based on the results presented above I conclude that the classifier is well balanced and can discriminate cars vs. other objects with high precision. As the next step I will use this classifier to detect and localise vehicles in images and video.

### Sliding Window Search

#### 1. Describe how you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

Car position and size can vary significantly across the image. Therefore, I applied the classifier over the image at various scales. However, the increase in the number of scales led to the increase in compute time need to process file. Through experimenting with various number of scales, the best performance was achieved by limiting the sliding window search to 3 scales: [1.5, 2.0, 2.5, 3.0]

In addition, to optimize the computation I limited the window search to bottom part of the image starting at Y=400. The bottom side of the window search was limited by the following formula:

ystop = ystart + scale * 64 * 1.25

The code for sliding window is presented in lines 46-79 (*'ClassifierTrainingPipeline.py'*).

An examples of a test image demonstraing how the pipeline works is presented below:
![alt text][SlidingWindow]

---

### Video Implementation

#### 1. Link to the final video output is presented below:
Here's a [link to my video result](./test_images/bbox_processed_video.mp4)


#### 2. Implementation of filters for false positives and method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

The code for filtering false positives and labeling boudnign boxes are presented in lines 72 and 74 respectively.



---

### Discussion

#### Discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Current approach delivers a good starting point. However, this approach is prone to errors in cases of objects occlusion. In addition, this pipelines fails in cases of small size objects.

As the next improvements to this projects is to apply latest advances in convolutional neural networks such as SSD or SqueezeNet (YOLO).

