## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./files/corner-12.jpg "Corners"
[image2]: ./files/calibration16.jpg "Squares"
[image3]: ./files/unwarped-1.jpg "Unwarped1"
[image4]: ./files/test5.jpg "image5"
[image5]: ./files/preprocess-1.jpg "Binary"
[image6]: ./files/warped-tracked-4.jpg "Warped"
[image7]: ./files/windows-tracked-5.jpg "Warped"
[image8]: ./files/first_frame_old_soln.jpg "Lane lines"
[image9]: ./errors/23sec_processed.jpg "Error 1"
[image10]: ./errors/ff.jpg "Error 2"
[video1]: ./output_tracked6.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines #5 through #49 of the file called `solution.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The `imgpoints` are chessboard corners that were found using `cv2.findChessboardCorners()` method.

![alt text][image2] ![alt text][image1]

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original image:
![test image 5][image4]
Unwarped image:
![unwarped][image3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![test image 5][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #61 through #112 in `solution.py`).  Here's an example of my output for this step.

![binary][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is a part of the `pipeline()` function, which appears in lines 124 through 155 in the file `solution.py`. I used the constants in the walkthrough, which worked for me as well:

```python
    src = np.float32([
        [img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct],
        [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct],
        [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim],
        [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim],
    ])
    # map to a rectangle
    offset = img.shape[1]*.25
    dst = np.float32([
        [offset, 0],
        [img.shape[1]-offset, 0],
        [img.shape[1]-offset, img.shape[0]],
        [offset, img.shape[0]]
    ])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Used the convolution approach. Sliding the window and measuring overlap. 

![alt text][image7]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #276 through #279 in my code in `solution.py` using the formula mentioned in the lectures.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #260 through #292 in my code in `solution.py` in the function `process_img()`.  Here is an example of my result on a test image:


![alt text][image8]
(Debug image, please ignore the radius of curvature text)


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_tracked6.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Was warping twice leading to this error:

![alt][image10]


Lighting changes were tough:

![alt][image9]

Had to tweak color threshholds. Added smoothing.



