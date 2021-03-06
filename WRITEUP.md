## Advanced Lane Finding
A pipeline to identify the lane boundaries in an image from a front-facing camera on a car.

**The project goals:**

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)
[image1]: ./camera_cal/calibration1.jpg "Distorted Chessboard"
[image2]: ./camera_cal/calibration1_undistorted.jpg "Undistorted Chessboard"
[image3]: ./test_images/test1.jpg "Test Image"
[image4]: ./output_images/test1_undistorted.jpg "Undistorted Test Image"
[image5]: ./output_images/test1_binary.jpg "Binary Test Image"
[image6]: ./output_images/test1_warped.jpg "Warped Test Image"
[image7]: ./output_images/test1_output.jpg "Output Test Image"


### Camera Calibration

The code for this step is contained in the following script: `./camera_cal/camera_calibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

Distorted Chessboard: ![alt text][image1]
Undisorted Chessboard: ![alt text][image2]


### Pipeline (single images)
The code for this pipeline is contained in the following script: `video_gen.py`.

#### 1. Apply distortion correction to raw images.

The code for this step is contained in the following script: `video_gen.py`. Reading the saved camera matrix and distortion coefficients between lines: `8-10`. Distortion correction is applied to input images on line: `75`.

Test Image: ![alt text][image3]
Undistorted Test Image: ![alt text][image4]


#### 2. Use color transforms, gradients, etc., to create a thresholded binary image.

The code for this step is contained in the following script: `video_gen.py`. I used a combination of color and gradient thresholds to generate a binary image. Thresholding functions (`abs_sobel_thresh()` and `color_threshold()`) are defined between lines: `14-51`. These thresholds are applied to undistorted images between lines: `79-83`.

Binary Test Image: ![alt text][image5]

#### 3. Apply perspective transform to rectify binary image ("birds-eye view").

The code for this step is contained in the following script: `video_gen.py`. The perspective transform is applied to binary images between lines: `87-113`.

I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[img_size[0] * (1 - top_trap_width) / 2, img_size[1] * trap_height],
     [img_size[0] * (1 + top_trap_width) / 2, img_size[1] * trap_height],
     [img_size[0] * (1 + bottom_trap_width) / 2, img_size[1] * bottom_img_trim],
     [img_size[0] * (1 - bottom_trap_width) / 2, img_size[1] * bottom_img_trim]])

offset = img_size[0] / 4

dst = np.float32(
    [[offset, 0],
     [img_size[0] - offset, 0],
     [img_size[0] - offset, img_size[1]],
     [offset, img_size[1]]])
```

Warped Test Image: ![alt text][image6]


#### 4. Detect lane pixels and fit to find the lane boundary.

The code for this step is contained in the following scripts: `video_gen.py` and `tracker.py`. The lanes are detected using convolution between lines: `116-200`.

My approach is to break image into 9 vertical layers, detect left and right lane pixels (by convolving the vertical image slices with the window template), fit a second order polynomial to left and right lanes, and the draw lanes.


#### 5. Determine the curvature of the lane and vehicle position with respect to center.

The code for this step is contained in the following script: `video_gen.py`. The curvature of the lane and vehicle position with respect to center is determined between lines: `210-231`.


#### 6. Warp the detected lane boundaries back onto the original image.

The code for this step is contained in the following script: `video_gen.py`. The lane boundaries are warped back on original image between lines: `202-203`.


#### 7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The code for this step is contained in the following script: `video_gen.py`. The numerical estimation of lane curvature and vehicle position are added to images between lines: `234-237`.

Output Test Image: ![alt text][image7]


### Pipeline (video)

The output project video (`./project_video_output.mp4`) identifies the lane boundaries in input project video (`./project_video.mp4`). Here is my [output video](./project_video_output.mp4)


### Discussion

**Drawbacks of current solution:**
* Based on road conditions, it challenging to detect and estimate the position of lanes. Current research suggests that we can get better results with deep learning techniques as an alternative to Computer Vision techniques used in the current implementation. A Convolutional Neural Network (CNN) based solution that yields an estimate of the lane position with respect to the vehicle from raw input image.
* Use of single front-facing camera. While driving, off-center shifts can be captured better with left and right cameras in addition to front-facing center camera.

**References:**
* DeepLanes: End-To-End Lane Position Estimation using Deep Neural Networks, by Alexandru Gurghian et al.
* End-to-End Deep Learning for Self-Driving Cars, by Mariusz Bojarski et al.
