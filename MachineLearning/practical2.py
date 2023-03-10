
# DataCampp Machine Learning for Iamge data, practical 2, image pre-processing and feature extraction

# PART1: image pre-processing

# Step 1: import the data

import glob
img_list = glob.glob('FlowerData/*.jpg')

print(img_list)
print(type(img_list))
print(type(img_list[0]))
print('directory has', len(img_list),'images')

# Step 2: display the image sizes

import cv2
for im_name in img_list:
    img = cv2.imread(im_name)
    print(img.shape)

# Step 3: resize an image and display along with original

nrow = 128
ncol = 128
col_features = nrow*ncol*3
print('each colour image will have',col_features,'features')
img = cv2.imread(img_list[13]) # an example image from our list
cv2.imshow('original image',img)
img_resized = cv2.resize(img,(nrow,ncol))
cv2.imshow('colour resized',img_resized); cv2.waitKey(0)

# Step 4: import as grayscale, resize, and display 

gray_features = nrow*ncol
print('each grayscale image will have',gray_features,'features')
img = cv2.imread(img_list[113], 0) # another example image from our list
cv2.imshow('original image',img)
img_resized = cv2.resize(img,(nrow,ncol))
cv2.imshow('resized image',img_resized); cv2.waitKey(0)

# Step 5: import all the images as a features array

import numpy as np
nrow = 128
ncol = 128
n_features = nrow * ncol
n_samples = len(img_list)
X = np.zeros((n_samples,n_features))

for i,im_name in enumerate(img_list):

    img = cv2.imread(im_name,0)

    # resize/reshape
    img_resized = cv2.resize(img,(nrow,ncol))

    # flatten into array
    img_flat = img_resized.flatten()

    X[i,:] = img_flat

print(X.shape)

# PART2: feature extraction

# Step 1: pick and convert an image

image = cv2.imread(img_list[13])

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Step 2: extract HOG features

from skimage.feature import hog
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1),visualize=True, channel_axis=-1)

from skimage import exposure
hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0, 10))



# Step 3: display HOG features next to original image

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4),sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# Step 4: compare number of features of HOG features to original image

print('original image has shape',image.shape,'and so',image.shape[0]*image.shape[1],'pixels')

print('hog representation instead has',fd.shape,'features')
