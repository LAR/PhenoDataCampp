# DataCampp machine learngin for image data, practical week 3, building an image classifier


# Step 1: Importing and formatting the data.

# Making the features matrix

# first import the required packages
import glob
import cv2
import numpy as np

# get a list of files with the .jpg extension from your flower data folder
img_list = glob.glob('FlowerData/*.jpg')

# set the number of rows and columns to 128
nrow = 128
ncol = 128

# calculate the number of features (rows x columns x 3 colour channels)
n_features = nrow * ncol * 3

# find the number of samples (how many images?)
n_samples = len(img_list)

# Make a matrix full of zeros with n_samples number of rows
# and n_features numbers of columns
X = np.zeros((n_samples,n_features))

# loop through the list of files
for i,im_name in enumerate(img_list):
    # open the image using cv2 (OpenCV)
    img = cv2.imread(im_name)

    # resize the image to be nrow by ncol pixels
    img_resized = cv2.resize(img,(nrow,ncol))

    # flatten the image data into a 1D array
    img_flat = img_resized.flatten()
    
    # add the flattened data to the ith row of matrix X
    X[i,:] = img_flat



# Making the target vector

species_list = ['Daffodil','Tigerlily','Tulip','Daisy']
y=[]
for species in species_list:
    y=y+[species]*80


# Step 2: Naive Bayes classifier - pixel data

# split the dataset
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

# intialise the classfier
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# train the model
model.fit(Xtrain, ytrain)


# evaluate the model
ymodel = model.predict(Xtest)

from sklearn.metrics import accuracy_score
score = accuracy_score(ytest, ymodel)
print(score)

# make and display the confusion matrix
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, ymodel)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=mat,display_labels=species_list)
disp.plot()

plt.show()

# Step 3: Naive bayes classifier - HOG features

import glob
import cv2
import numpy as np
img_list = glob.glob('FlowerData/*.jpg')
nrow = 128
ncol = 128

pix_per_cell = 16
orients = 8
# calculate the number of features, making sure it returns an integer
n_features = int((nrow/pix_per_cell) * (ncol/pix_per_cell) * orients)
print('number of HOG features should be',n_features)

from skimage.feature import hog
# Make the empty features matrix
X = np.zeros((n_samples,n_features))

# loop through the list of files
for i,im_name in enumerate(img_list):
    # open the image
    img = cv2.imread(im_name)
    # resize the image
    img_resized = cv2.resize(img,(nrow,ncol))
    # get the HOG features
    img_hog = hog(img_resized, orientations=orients,pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(1, 1),visualize=False, channel_axis=-1)
    # add the data to the ith row of matrix X
    X[i,:] = img_hog

# split the dataset
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

# intialise the classfier
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# train the model
model.fit(Xtrain, ytrain)


# evaluate the model
ymodel = model.predict(Xtest)

from sklearn.metrics import accuracy_score
score = accuracy_score(ytest, ymodel)
print(score)

# Step 4 (challenge): Other Classifier Methods

# to do this you can use the code as for the Naive Bayes,
# but just change the two lines where indicated

import glob
import cv2
import numpy as np


img_list = glob.glob('FlowerData/*.jpg')


nrow = 128
ncol = 128


n_features = nrow * ncol * 3


n_samples = len(img_list)

X = np.zeros((n_samples,n_features))


for i,im_name in enumerate(img_list):
    img = cv2.imread(im_name)

    img_resized = cv2.resize(img,(nrow,ncol))

    img_flat = img_resized.flatten()
    
    X[i,:] = img_flat



species_list = ['Daffodil','Tigerlily','Tulip','Daisy']
y=[]
for species in species_list:
    y=y+[species]*80


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=7)

# change the algorithm HERE
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()

model.fit(Xtrain, ytrain)


ymodel = model.predict(Xtest)

from sklearn.metrics import accuracy_score
score = accuracy_score(ytest, ymodel)
print(score)


from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, ymodel)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=mat,display_labels=species_list)
disp.plot()

plt.show()
