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


**Camera calibration:**
The script (`camera_calibration.py`), images, and output (`calibration_pickle.p`) for camera calibration are stored in the `camera_cal` directory.

**Test images:**
The images in `test_images` directory are for testing the pipeline on single frames. The script (`image_gen.py`) is used to generate output from each stage of the pipeline for all the images in `test_images`. These output images are saved in `output_images` directory.

For example, `test1.jpg`, has corresponding output images:
* `test1_undistorted.jpg` -- Distortion corrected `test1.jpg`
* `test1_binary.jpg` -- Binary `test1.jpg`
* `test1_warped.jpg` -- Perspective transformed `test1.jpg`
* `test1_output.jpg` -- Output with lane boundaries on `test1.jpg`

**Project video:**
The script (`video_gen.py`) is used to apply the pipeline to identify the lane boundaries in input video (`project_video.mp4`), and generate the output video (`project_video_output.mp4`).


## Implementation Details

Please see WRITEUP.md
