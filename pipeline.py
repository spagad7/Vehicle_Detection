import cv2
import pickle
import argparse
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from detect import *

def pipeline():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('v',
                        type=str,
                        help="path to video file")
    args = parser.parse_args()

    # Load trained model from pickled file
    f = open("models/model_svc.p", "rb")
    settings = pickle.load(f)
    f.close()

    # Create detect class instance
    d = Detector(settings)

    '''
    img = mpimg.imread(args.v)
    #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_detect = d.detect(img, win_size=64, y_start=400, y_stop=656)
    cv2.imshow("Output", img_detect)
    cv2.waitKey(0)
    '''

    # Process frames
    video = VideoFileClip(args.v)
    cv2.namedWindow("Output")
    for frame in video.iter_frames():
        # moviepy's default colospace is RGB
        # opencv's default colorspace is BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_detect = d.detect(frame_bgr, y_start=400, y_stop=656)
        cv2.imshow("Output", frame_detect)
        cv2.waitKey(1)


if __name__=='__main__': pipeline()
