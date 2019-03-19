from moviepy.editor import VideoFileClip
import pickle
import cv2
import numpy as np
from tracker import Tracker

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("./camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


# Identify pixels where the gradient of an image falls within a specified threshold range.
def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gradient in the x-direction
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # Gradient in the y-direction
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_img = np.zeros_like(scaled_sobel)
    binary_img[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_img


def color_threshold(img, s_thresh=(0, 255), v_thresh=(0, 100)):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    # Use exclusive lower bound (>) and inclusive upper bound (<=)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    # Use exclusive lower bound (>) and inclusive upper bound (<=)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    color_binary = np.zeros_like(s_channel)
    color_binary[(s_binary == 1) & (v_binary == 1)] = 1

    return color_binary


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    # NOTE_AV: Modified from lecture, using width instead of width/2 (src: sliding_window_convolution.py)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
           max(0, int(center - width)):min(int(center + width), img_ref.shape[1])] = 1
    return output


def process_image(img):
    """ Pipeline to identify the lane boundaries in an image from a front-facing
    camera on a car.

        [1] Distortion Correction
        [2] Color/Gradient Threshold
        [3] Perspective Transform
        [4] Detect Lane Lines using Convolution
        [5] Determine the lane curvature
    """

    # *** [1] DISTORTION CORRECTION ***
    # Undistorting an image using mtx and dist
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # *** [2] COLOR/GRADIENT THRESHOLD ***
    # Generate binary image
    binary_img = np.zeros_like(img[:, :, 0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(24, 255))
    color_binary = color_threshold(img, s_thresh=(100, 255), v_thresh=(50, 255))
    binary_img[(gradx == 1) & (grady == 1) | (color_binary == 1)] = 255

    # *** [3] PERSPECTIVE TRANSFORM ***
    # Perspective transform area
    img_size = (img.shape[1], img.shape[0])    # (width, height)
    bottom_trap_width = 0.76    # Percentage of trapezoid bottom width
    top_trap_width = 0.08    # Percentage of trapezoid top width
    trap_height = 0.62    # Percentage of trapezoid height (from top?)
    bottom_img_trim = 0.935    # Percentage to image to eliminate car's hood

    # NOTE_AV: Modified from lecture (src: undistort_transform.py)
    src = np.float32([[img_size[0] * (1 - top_trap_width) / 2, img_size[1] * trap_height],
                      [img_size[0] * (1 + top_trap_width) / 2, img_size[1] * trap_height],
                      [img_size[0] * (1 + bottom_trap_width) / 2, img_size[1] * bottom_img_trim],
                      [img_size[0] * (1 - bottom_trap_width) / 2, img_size[1] * bottom_img_trim]])

    # Offset for dst points
    offset = img_size[0] / 4

    # NOTE_AV: Modified from lecture (src: undistort_transform.py)
    # dst = np.float32([[offset, offset], [img_size[0] - offset, offset], [img_size[0] - offset, img_size[1] - offset],
    #                  [offset, img_size[1] - offset]])
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]],
                      [offset, img_size[1]]])

    # Compute the perspective transform, M, given source and destination points
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse perspective transform, Minv, given source and destination points
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp an image using the perspective transform, M
    warped_img = cv2.warpPerspective(binary_img, M, img_size)

    # *** [4] DETECT LANE LINES USING CONVOLUTION ***
    window_width = 25
    window_height = 80    # Break image into 9 vertical layers since image height is 720
    margin = 25
    xm = 4 / 384
    ym = 10 / 720
    smooth_factor = 15

    curve_centers = Tracker(
        win_width=window_width, win_height=window_height, margin=margin, xm=xm, ym=ym, smooth_factor=smooth_factor)

    window_centroids = curve_centers.find_window_centroids(warped_img)

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped_img)
    r_points = np.zeros_like(warped_img)

    # Create empty lists to receive left and right lane pixel
    leftx = []
    rightx = []

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height, warped_img, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped_img, window_centroids[level][1], level)
        # Add graphic points from window mask here to total pixels found
        # l_points[(l_points == 255) | ((l_mask == 1))] = 255
        # r_points[(r_points == 255) | ((r_mask == 1))] = 255
        l_points[(l_mask == 1)] = 255
        r_points[(r_mask == 1)] = 255

    # Draw the results
    # Add both left and right window pixels together
    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)    # create a zero color channel
    # Make window pixels green
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    # Making the original road pixels 3 color channels
    warped_color_img = np.dstack((warped_img, warped_img, warped_img)) * 255
    # Overlay the orignal road image with window results
    output_img = cv2.addWeighted(warped_color_img, 1, template, 0.5, 0.0)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_size[1] - 1, img_size[1])

    # What does result_yvals mean?
    # NOTE: result_yvals is not a np.array()
    result_yvals = np.arange(img_size[1] - window_height / 2, 0, -window_height)

    # Fit a second order polynomial to left lane
    # NOTE_AV: Instead of np.array(result_yvals, np.float32), project walk-through uses result_yvals
    # NOTE_AV: Instead of np.array(leftx, np.float32), project walk-through uses leftx
    left_fit = np.polyfit(np.array(result_yvals, np.float32), np.array(leftx, np.float32), 2)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]

    # Fit a second order polynomial to right lane
    # NOTE_AV: Instead of np.array(result_yvals, np.float32), project walk-through uses result_yvals
    # NOTE_AV: Instead of np.array(rightx, np.float32), project walk-through uses rightx
    right_fit = np.polyfit(np.array(result_yvals, np.float32), np.array(rightx, np.float32), 2)
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Draw lanes
    left_lane = np.array(
        list(
            zip(
                np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
                np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)
    right_lane = np.array(
        list(
            zip(
                np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)
    inner_lane = np.array(
        list(
            zip(
                np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] - window_width / 2), axis=0),
                np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

    road_img = np.zeros_like(img)
    road_back_img = np.zeros_like(img)
    cv2.fillPoly(road_img, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road_img, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road_img, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_back_img, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_back_img, [right_lane], color=[255, 255, 255])

    road_warped_img = cv2.warpPerspective(road_img, Minv, img_size)
    road_warped_back_img = cv2.warpPerspective(road_back_img, Minv, img_size)

    base_img = cv2.addWeighted(img, 1.0, road_warped_back_img, -1.0, 0.0)
    output_img = cv2.addWeighted(base_img, 1.0, road_warped_img, 0.7, 0.0)

    # *** [5] DETERMINE THE LANE CURVATURE ***
    # Curvature of left lane
    xm_per_pix = curve_centers.xm_per_pix
    ym_per_pix = curve_centers.ym_per_pix

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit a first order polynomial to find curvature of left lane
    # NOTE_AV: leftx is a list, need to convert to np.array()
    # NOTE_AV: Instead of ploty, project walk-through uses np.array(result_yvals, np.float32)
    left_fit_cr = np.polyfit(
        np.array(result_yvals, np.float32) * ym_per_pix,
        np.array(leftx, np.float32) * xm_per_pix, 2)
    left_curverad = (
        (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])

    # Calculate offset of car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - img_size[0] / 2) * xm_per_pix
    camera_pos = 'left'
    if center_diff <= 0:
        camera_pos = 'right'

    # Add text showing curvature and offset to output image
    cv2.putText(output_img, 'Radius of curvature is ' + str(round(left_curverad, 3)) + 'm.', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output_img, 'Car is ' + str(abs(round(center_diff, 3))) + 'm ' + camera_pos + ' from lane center.',
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return output_img


input_video = input('Enter input video filename: ')
output_video = input_video.split('.')[0] + '_output.mp4'

# Identify the lane boundaries in an input_video from a front-facing camera on a
# car and save an output_video.
input_clip = VideoFileClip(input_video)
output_clip = input_clip.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)
