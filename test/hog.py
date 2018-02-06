import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread("test_images/test8.png")
orient=9
px=8
cell_blk=2
norm='L2-Hys'
vis=True
transform=True
feat_vec=True

img_ch = img[:,:,0]

features, img_hog = hog(img_ch, orientations=orient,
                        pixels_per_cell=(px, px),
                        cells_per_block=(cell_blk, cell_blk),
                        block_norm=norm,
                        visualise=vis,
                        transform_sqrt=transform,
                        feature_vector=feat_vec)


# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(img_ch, cmap='gray')
plt.title('Example Non Car Image')
plt.subplot(122)
plt.imshow(img_hog, cmap='gray')
plt.title('HOG Visualization')
plt.show()
