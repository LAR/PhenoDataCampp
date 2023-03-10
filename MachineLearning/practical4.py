# DataCampp machine learning for image data, practical 4: dta augmentation

# Part 1: data augmentation

# get original image list

import glob
img_list = glob.glob('FlowerData/*.jpg')

# display an example rotated image

import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import rotate
import numpy as np
img = imread(img_list[13])
random_angle = np.random.uniform(15,45) # get a random angle
rotated = rotate(img,random_angle)
plt.imshow(rotated)
plt.axis('off')
plt.show()

# loop through and a save a randomly rotated image in the folder 'rotations'
# NOTE: you need to make this folder first

for i,im_name in enumerate(img_list):
    img = imread(im_name)
    random_angle = np.random.uniform(15,45)
    rotated = rotate(img,random_angle) # gives floats between 0 and 1
    rotated = (255*rotated).astype(np.uint8) # convert back to 0-255 ints
    imsave('rotations/rotated_%03d.jpg' % i, rotated)

# loop through and a save a left-right flipped image in the folder 'flips'
# NOTE: you need to make this folder first

for i,im_name in enumerate(img_list):
    img = imread(im_name)
    flipped = np.fliplr(img)
    imsave('flips/flipped_%03d.jpg' % i, flipped)

# Part 2: Training a model using the augmented dataset

# make expanded features matrix with augmented dataset

import cv2


nrow = 128
ncol = 128

n_features = nrow * ncol * 3

orig_list = glob.glob('FlowerData/*.jpg')
flip_list = glob.glob('flips/*.jpg')
rotation_list = glob.glob('rotations/*.jpg')

img_list = orig_list + flip_list + rotation_list

n_samples = len(img_list)

X = np.zeros((n_samples,n_features))
for i,im_name in enumerate(img_list):
    img = cv2.imread(im_name)
    # resize/reshape
    img_resized = cv2.resize(img,(nrow,ncol))

    # flatten into array
    img_flat = img_resized.flatten()

    X[i,:] = img_flat

# make exapnded target vector with the augmented dataset

species_list = ['Daffodil','Tigerlily','Tulip','Daisy']
y=[]
for species in species_list:
    y=y+[species]*80

y=y+y+y

y = np.array(y)

# train and test the model

# split into training / validation
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)


# choose / set up model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# train model
model.fit(Xtrain, ytrain)

y_model = model.predict(Xtest)

# evaluate accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(ytest, y_model)
print(score)

# Part 3: cross-validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10)
print(scores)

print(np.mean(scores))

