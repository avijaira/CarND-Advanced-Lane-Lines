from moviepy.editor import VideoFileClip
import pickle
import cv2
import numpy as np
import glob
from tracker import Tracker

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("./camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


def process_image(img):

    return output_img


output_video = 'project_output_video.mp4'
input_video = 'project_video.mp4'

clip = VideoFileClip(input_video)
video_clip = clip.f1_image(process_image)
video_clip.write_videofile(output_video, audio=False)
