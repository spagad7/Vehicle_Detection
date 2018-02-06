import cv2
import pickle
import argparse
import numpy as np
import matplotlib.image as mpimg
from collections import deque
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from detect import *
import matplotlib.pyplot as plt


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def get_labels(img, windows):
    img_out = np.copy(img)
    img_heat = np.zeros_like(img_out[:,:,0])
    # Generate heat image
    for window in windows:
        top = window[0][1]
        bottom = window[1][1]
        left = window[0][0]
        right = window[1][0]
        img_heat[top:bottom, left:right] += 1

    # Threshold heat image
    thresh = 15
    img_heat[img_heat <= thresh] = 0
    labels = label(img_heat)

    return labels


def pipeline():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('v',
                        type=str,
                        help="path to video file")
    args = parser.parse_args()

    # Load trained model from pickled file
    f = open("models/model_svc_hsv.p", "rb")
    settings = pickle.load(f)
    f.close()

    # Create detect class instance
    d = Detector(settings)

    # Process frames
    video = VideoFileClip(args.v)
    #cv2.namedWindow("Heat")
    cv2.namedWindow("Output")

    # List to maintain window history
    win_history = deque()
    history_ct = 10
    i = 0
    for frame in video.iter_frames():
        # List to store detected window
        windows = []

        win1 = d.detect(frame, scale=1, y_start=400, y_stop=496)
        if len(win1) != 0:
            windows.append(win1)

        win2 = d.detect(frame, scale=1.5, y_start=400, y_stop=544)
        if len(win2) != 0:
            windows.append(win2)

        win3 = d.detect(frame, scale=2, y_start=400, y_stop=592)
        if len(win3) != 0:
            windows.append(win3)

        win4 = d.detect(frame, scale=3.5, y_start=400, y_stop=656)
        if len(win4) != 0:
            windows.append(win4)

        if len(windows) != 0:
            win_list = np.concatenate(windows)
            # Update window history
            if len(win_history) < history_ct:
                win_history.append(win_list)
            elif len(win_history) == history_ct:
                win_history.popleft()
                win_history.append(win_list)
            elif len(win_history) > history_ct:
                win_history.pop_left()
            win_history_list = np.concatenate(win_history)
            labels = get_labels(frame, win_history_list)
            frame_label = draw_labeled_bboxes(frame, labels)
            # moviepy's default colorspace is RGB
            # opencv's default colorspace is BGR
            frame_out = cv2.cvtColor(frame_label, cv2.COLOR_RGB2BGR)
        else:
            frame_out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output", frame_out)
        filename = "output/img" + str(i) + ".png"
        i += 1
        cv2.imwrite(filename, frame_out)
        cv2.waitKey(1)


if __name__ == '__main__': pipeline()
