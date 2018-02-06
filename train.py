import os
import argparse
import cv2
import glob
import pickle
import numpy as np
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from features import *


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path',
                        type=str,
                        help="path to image file")
    args = parser.parse_args()

    # Get vehicle and non-vehicle image list
    img_vehicles = get_image_list(args.dataset_path + "/vehicles")
    img_non_vehicles = get_image_list(args.dataset_path + "/non-vehicles")

    settings = {'cspace': 'HSV',
                'spatial': False,
                'sp_img_size': (16, 16),
                'hist': False,
                'nbins': 32,
                'bins_range': (0, 256),
                'hog': True,
                'orientations': 8,
                'pixels_per_cell': 16,
                'cells_per_block': 2,
                'block_norm': 'L2-Hys',
                'visualize': False,
                'transform_sqrt': True,
                'feature_vector': True}

    # Get vehicle features
    feat_vehicles = []
    for img_path in img_vehicles:
        # Read image
        img = mpimg.imread(img_path)
        # Scale image to [0-255]
        img_scaled = np.uint8((img * 255) / np.max(img))
        # Get features
        feat_vehicles.append(get_features(img_scaled, settings))

    # Get non-vehicle features
    feat_non_vehicles = []
    for img_path in img_non_vehicles:
        # Read image
        img = mpimg.imread(img_path)
        # Scale image to [0-255]
        img_scaled = np.uint8((img * 255) / np.max(img))
        # Get features
        feat_non_vehicles.append(get_features(img_scaled, settings))

    # Create feature vector
    x = np.vstack((feat_vehicles, feat_non_vehicles)).astype(np.float64)
    # Scale features
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)

    # Create label vector
    y = np.hstack((np.ones(len(img_vehicles)), np.zeros(len(img_non_vehicles))))

    # Split training data into training and validation set
    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y,
                                                        test_size=0.2,
                                                        random_state=rand_state)

    # Train model
    svc = LinearSVC()
    svc.fit(x_train, y_train)
    print("Score = ", svc.score(x_test, y_test))
    # Save trained model
    settings['scaler'] = scaler
    settings['svc'] = svc
    f_name = "models/model_svc_" + settings['cspace'].lower() + ".p"
    f = open(f_name, "wb")
    pickle.dump(settings, f)
    print("Saved trained model in ", f_name)


# Function to get image names from dataset
def get_image_list(dataset_path):
    img_list = []
    # Check if the path has image files
    img_list.append(glob.glob(dataset_path + "/*.png"))
    # Check if the path has sub directories
    for fname in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, fname)):
            dir_path = os.path.join(dataset_path, fname)
            img_list.append(glob.glob(dir_path + "/*.png"))
    # Concatenate
    img_list = np.concatenate(img_list)
    return img_list


if __name__ == '__main__':
    train()
