# DataCampp Machine learning for image data, practical 5, Multilayer perceptron

# Part 1: MLP with digits


from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data
y = digits.target

print(X.shape)

# split into training / validation
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=13)


# choose / set up model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(64,),max_iter=500)


model.fit(Xtrain, ytrain)

y_model = model.predict(Xtest)


from sklearn.metrics import accuracy_score
score = accuracy_score(ytest, y_model)
print(score)

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=mat,
display_labels=digits.target_names)
disp.plot()

plt.show()



# Part 2: look at coeffs

print('the model has',len(model.coefs_),'sets of coefficients')
print('the first has shape',model.coefs_[0].shape)
print('the second has shape',model.coefs_[1].shape)





# Part 3 (optional) - try with flower image data

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

    # resize/reshape
    img_resized = cv2.resize(img,(nrow,ncol))

    # flatten into array
    img_flat = img_resized.flatten()

    X[i,:] = img_flat

species_list = ['Daffodil','Tigerlily','Tulip','Daisy']

y=[]

for species in species_list:
    y=y+[species]*80

# split into training / validation
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=13)


# choose / set up model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(64,64,),max_iter=500)



model.fit(Xtrain, ytrain)


y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
score = accuracy_score(ytest, y_model)
print(score)

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=mat,display_labels=species_list)
disp.plot()

plt.show()

