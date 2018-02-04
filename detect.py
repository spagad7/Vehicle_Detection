import cv2
import numpy as np
from features import *


class Detector():
    def __init__(self, settings):
        # Load settings from pickled data
        self.cspace = settings['cspace']
        # self.cspace = 'YCrCb'
        self.spatial = settings['spatial']
        self.sp_img_size = settings['sp_img_size']
        self.hist = settings['hist']
        self.nbins = settings['nbins']
        self.bins_range = settings['bins_range']
        self.hog = settings['hog']
        self.orient = settings['orientations']
        self.pix_per_cell = settings['pixels_per_cell']
        self.cell_per_blk = settings['cells_per_block']
        self.blk_norm = settings['block_norm']
        self.viz = settings['visualize']
        self.trans = settings['transform_sqrt']
        self.feat_vec = False
        self.scaler = settings['scaler']
        self.svc = settings['svc']
        self.scale = 1.5

    # Function to detect vehicles in image
    def detect(self, img, win_size=64, y_start=0, y_stop=1280):
        img_out = np.copy(img)
        # Convert colorspace of image
        if self.cspace != 'RGB':
            color = getattr(cv2, 'COLOR_RGB2' + self.cspace)
            img_cspace = cv2.cvtColor(img, color)
        else:
            img_cspace = img

        # Get ROI from input image
        img_crop = img_cspace[y_start:y_stop, :, :]

        imshape = img_crop.shape
        img_roi = cv2.resize(img_crop, (np.int(imshape[1] / self.scale), np.int(imshape[0] / self.scale)))

        # Get hog features of ROI
        hog_features = []
        if self.hog:
            for ch in range(img_cspace.shape[2]):
                hog_features.append(getHogFeat(img_roi[:, :, ch],
                                               orient=self.orient,
                                               px=self.pix_per_cell,
                                               cell_blk=self.cell_per_blk,
                                               norm=self.blk_norm,
                                               vis=self.viz,
                                               transform=self.trans,
                                               feat_vec=self.feat_vec))

        # Configure windows
        # Get number of blocks along x and y axis
        nx_blocks = (img_roi.shape[1] // self.pix_per_cell) - self.cell_per_blk + 1
        ny_blocks = (img_roi.shape[0] // self.pix_per_cell) - self.cell_per_blk + 1
        # Get number of blocks per window
        blk_per_win = (win_size // self.pix_per_cell) - self.cell_per_blk + 1
        # Get number of windows along the x and y axis
        # cell_step must be a multiple of number of cells per block
        cell_step = 2
        nx_win = (nx_blocks - blk_per_win) // cell_step + 1
        ny_win = (ny_blocks - blk_per_win) // cell_step + 1

        x_start = 0
        x_stop = img_roi.shape[1]

        '''
        print("hog_features.shape = ", hog_features[0].shape)
        print("nx_blocks = ", nx_blocks)
        print("ny_blocks = ", ny_blocks)
        print("blk_per_win = ", blk_per_win)
        print("nx_win = ", nx_win)
        print("ny_win = ", ny_win)
        '''

        # Slide windows, extracting HOG, spatial and color histogram features
        for x in range(nx_win):
            for y in range(ny_win):
                x_pos = x * cell_step
                y_pos = y * cell_step
                features_win = []
                # Get the subimage corresponding to the window
                left = x_pos * self.pix_per_cell
                top = y_pos * self.pix_per_cell
                right = left + win_size
                bottom = top + win_size
                img_win = cv2.resize(img_roi[top:bottom, left:right, :],
                                     (win_size, win_size))

                # Get spatial features
                if self.spatial:
                    spatial = getSpatialFeat(img_win, size=self.sp_img_size)
                    # print("spatial.shape = ", spatial.shape)
                    features_win.append(spatial)
                # Get color histogram features
                if self.hist:
                    hist = getHistFeat(img_roi, nbins=self.nbins,
                                       bins_range=self.bins_range)
                    # print("hist.shape = ", hist.shape)
                    features_win.append(hist)
                # Get hog features
                for i in range(img_cspace.shape[2]):
                    temp = hog_features[i][y_pos:y_pos + blk_per_win,
                           x_pos:x_pos + blk_per_win].ravel()
                    # print("temp.shape = ", temp.shape, " y_pos = ", y_pos)
                    features_win.append(temp)

                features = np.concatenate(features_win).reshape(1, -1)
                # print("features.shape = ", features.shape)
                features_scaled = self.scaler.transform(features)
                prediction = self.svc.predict(features_scaled)

                # If vehicle is detected
                if prediction == 1:
                    left_sc = np.int(left * self.scale)
                    top_sc = np.int(top * self.scale) + y_start
                    right_sc = np.int(right * self.scale)
                    bottom_sc = np.int(bottom * self.scale) + y_start
                    cv2.rectangle(img_out, (left_sc, top_sc),
                                  (right_sc, bottom_sc), (0, 0, 255), 6)

        return img_out
