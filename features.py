#!/usr/bin/python3

import numpy as np
import cv2
from skimage.feature import hog


# Function to get spatially binned features
def get_spatial_feat(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features


# Function to get color histogram features
def get_hist_feat(img, nbins=32, bins_range=(0, 256)):
    feat = []
    for ch in range(img.shape[2]):
        hist = np.histogram(img[:, :, ch], bins=nbins, range=bins_range)
        # hist a tuple of histogram values and bin_edges, we want values
        feat.append(hist[0])
    features = np.concatenate(feat)
    return features


# Function to get hog features
def get_hog_feat(img, orient=9, px=8, cell_blk=2, norm=True,
               vis=False, transform=True, feat_vec=True):
    if vis:
        features, img_hog = hog(img, orientations=orient,
                                pixels_per_cell=(px, px),
                                cells_per_block=(cell_blk, cell_blk),
                                block_norm=norm,
                                visualise=vis,
                                transform_sqrt=transform,
                                feature_vector=feat_vec)
        return features, img_hog
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(px, px),
                       cells_per_block=(cell_blk, cell_blk),
                       block_norm=norm,
                       visualise=vis,
                       transform_sqrt=transform,
                       feature_vector=feat_vec)
        return features


# Function to concatenate spatial, color-histogram and hog features
def get_features(img, settings):
    # Change color space
    if settings['cspace'] != 'RGB':
        cspace = getattr(cv2, "COLOR_RGB2" + settings['cspace'])
        img_cspace = cv2.cvtColor(img, cspace)
    else:
        img_cspace = img

    # Extract features
    features = []
    if settings['spatial']:
        features.append(get_spatial_feat(img_cspace, size=settings['sp_img_size']))
    if settings['hist']:
        features.append(get_hist_feat(img_cspace, nbins=settings['nbins'],
                                    bins_range=settings['bins_range']))
    if settings['hog']:
        for ch in range(img_cspace.shape[2]):
            features.append(get_hog_feat(img_cspace[:, :, ch],
                                       orient=settings['orientations'],
                                       px=settings['pixels_per_cell'],
                                       cell_blk=settings['cells_per_block'],
                                       norm=settings['block_norm'],
                                       vis=settings['visualize'],
                                       transform=settings['transform_sqrt'],
                                       feat_vec=settings['feature_vector']))

    features_all = np.concatenate(features)
    return features_all
