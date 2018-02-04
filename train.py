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
    img_vehicles = getImageList(args.dataset_path+"/vehicles")
    img_non_vehicles = getImageList(args.dataset_path+"/non-vehicles")

    settings = {'cspace':'YCrCb',
                'spatial':True,
                'sp_img_size':(32, 32),
                'hist':True,
                'nbins':32,
                'bins_range':(0, 256),
                'hog':True,
                'orientations':9,
                'pixels_per_cell':8,
                'cells_per_block':2,
                'block_norm':'L2-Hys',
                'visualize':False,
                'transform_sqrt':True,
                'feature_vector':True}

    # Get vehicle features
    feat_vehicles = []
    for img_path in img_vehicles:
        # Read image
        img = mpimg.imread(img_path)
        # Scale image to [0-255]
        img_scaled = np.uint8((img*255)/np.max(img))
        # Get features
        feat_vehicles.append(getFeatures(img_scaled, settings))

    # Get non-vehicle features
    feat_non_vehicles = []
    for img_path in img_non_vehicles:
        # Read image
        img = mpimg.imread(img_path)
        # Scale image to [0-255]
        img_scaled = np.uint8((img*255)/np.max(img))
        # Get features
        feat_non_vehicles.append(getFeatures(img_scaled, settings))

    # Create feature vector
    x = np.vstack((feat_vehicles, feat_non_vehicles)).astype(np.float64)
    # Scale features
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)

    # Create label vector
    y = np.hstack((np.ones(len(img_vehicles)), np.zeros(len(img_non_vehicles))))

    # Split training data into training and validation set
    rand_state = np.random.randint(0,100)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y,
                                                    test_size=0.2,
                                                    random_state = rand_state)

    # Train model
    svc = LinearSVC()
    svc.fit(x_train, y_train)
    print("Score = ", svc.score(x_test, y_test))
    # Save trained model
    save_dict = {'cspace':'YCrCb',
                'spatial':True,
                'sp_img_size':(32, 32),
                'hist':True,
                'nbins':32,
                'bins_range':(0, 256),
                'hog':True,
                'orientations':9,
                'pixels_per_cell':8,
                'cells_per_block':2,
                'block_norm':'L2-Hys',
                'visualize':False,
                'transform_sqrt':True,
                'feature_vector':True,
                'scaler':scaler,
                'svc':svc}
    f = open("models/model_svc.p", "wb")
    pickle.dump(save_dict, f)
    print("Saved model in models/model_svc.p")



# Function to get image names from dataset
def getImageList(dataset_path):
    img_list = []
    # Check if the path has image files
    img_list.append(glob.glob(dataset_path+"/*.png"))
    # Check if the path has sub directories
    for fname in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path,fname)):
            dir_path = os.path.join(dataset_path,fname)
            img_list.append(glob.glob(dir_path+"/*.png"))
    # Concatenate
    img_list = np.concatenate(img_list)
    return img_list





if __name__=='__main__': train()
