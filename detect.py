import cv2
import numpy as np
from features import *


class Detector:
    def __init__(self, settings):
        # Load settings from pickled data
        self.cspace = settings['cspace']
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
        self.win_size = 64

    # Function to detect vehicles in image
    def detect(self, img, scale=1.0, y_start=0, y_stop=1280):
        img_out = np.copy(img)
        # Convert colorspace of image
        if self.cspace != 'RGB':
            color = getattr(cv2, 'COLOR_RGB2' + self.cspace)
            img_cspace = cv2.cvtColor(img, color)
        else:
            img_cspace = img

        # Get ROI from input image and scale it
        img_crop = img_cspace[y_start:y_stop, :, :]
        imgshape = img_crop.shape
        img_scaled = cv2.resize(img_crop, (np.int(imgshape[1] / scale),
                                           np.int(imgshape[0] / scale)))

        # Get hog features of ROI
        hog_features = []
        if self.hog:
            for ch in range(img_cspace.shape[2]):
                hog_features.append(get_hog_feat(img_scaled[:, :, ch],
                                               orient=self.orient,
                                               px=self.pix_per_cell,
                                               cell_blk=self.cell_per_blk,
                                               norm=self.blk_norm,
                                               vis=self.viz,
                                               transform=self.trans,
                                               feat_vec=self.feat_vec))

        # Configure windows
        # Get number of blocks along x and y axis
        nx_blocks = ((img_scaled.shape[1] // self.pix_per_cell)
                    - self.cell_per_blk + 1)
        ny_blocks = ((img_scaled.shape[0] // self.pix_per_cell)
                    - self.cell_per_blk + 1)
        # Get number of blocks per window
        blk_per_win = ((self.win_size // self.pix_per_cell)
                      - self.cell_per_blk + 1)
        # Get number of windows along the x and y axis
        # cell_step must be a multiple of number of cells per block
        cell_step = 2
        nx_win = (nx_blocks - blk_per_win) // cell_step + 1
        ny_win = (ny_blocks - blk_per_win) // cell_step + 1

        # List for storing detected windows
        windows = []
        # Slide windows, extracting HOG, spatial and color histogram features
        for x in range(nx_win):
            for y in range(ny_win):
                x_pos = x * cell_step
                y_pos = y * cell_step
                features_win = []
                # Get the subimage corresponding to the window
                left = x_pos * self.pix_per_cell
                top = y_pos * self.pix_per_cell
                right = left + self.win_size
                bottom = top + self.win_size
                img_win = cv2.resize(img_scaled[top:bottom, left:right, :],
                                     (self.win_size, self.win_size))

                # Get spatial features
                if self.spatial:
                    spatial = get_spatial_feat(img_win, size=self.sp_img_size)
                    features_win.append(spatial)
                # Get color histogram features
                if self.hist:
                    hist = get_hist_feat(img_scaled, nbins=self.nbins,
                                       bins_range=self.bins_range)
                    features_win.append(hist)
                # Get hog features
                for i in range(img_cspace.shape[2]):
                    temp = hog_features[i][y_pos:y_pos + blk_per_win,
                           x_pos:x_pos + blk_per_win].ravel()
                    features_win.append(temp)

                features = np.concatenate(features_win).reshape(1, -1)
                # print("features.shape = ", features.shape)
                features_scaled = self.scaler.transform(features)
                prediction = self.svc.predict(features_scaled)

                # If vehicle is detected
                if prediction == 1:
                    left_sc = np.int(left * scale)
                    top_sc = np.int(top * scale) + y_start
                    right_sc = np.int(right * scale)
                    bottom_sc = np.int(bottom * scale) + y_start
                    windows.append(((left_sc, top_sc), (right_sc, bottom_sc)))
                    # cv2.rectangle(img_out, (left_sc, top_sc),
                    #              (right_sc, bottom_sc), (0, 0, 255), 6)

        return windows
