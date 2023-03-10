# DataCampp Machine Learning for image data, practical 1

# STEP 1: Importing the data

# import the data using sklearn
from sklearn.datasets import load_iris
iris = load_iris()
print(type(iris))

# show some information about what exactly you have imported
print(iris.keys())

# show feature names and shape of matrix
print(iris.feature_names,iris.data.shape)
# show target names and shape of matrix
print(iris.target_names,iris.target.shape)

X = iris.data
y = iris.target


# ASIDE : Pandas DataFrame

# import again using as_frame=True
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame

# display some information
print(type(df))

print(df.head())


# extract data from frame in desired format
X = df.drop('target', axis=1)
print(X.head())

y = df['target']


# STEP 2: visualising the data

import matplotlib.pyplot as plt
xy_plot = plt.scatter(X['sepal width (cm)'],X['petal width (cm)'],c=y)
plt.xlabel('sepal width (cm)')
plt.ylabel('petal width (cm)')
plt.legend(handles=xy_plot.legend_elements()[0],labels=list(iris.target_names))
plt.show()

# STEP 3: Training a simple machine learning model

# import and initialise the KNN classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)

# split the dataset
from sklearn.model_selection import train_test_split
Xtrain, Xtest, y_train, y_test = train_test_split(X, y ,random_state=13)

# train the model
model.fit(Xtrain,y_train)


# STEP 4: Evaluating the results

from sklearn.metrics import accuracy_score

# make predections with the test data
y_predicted = model.predict(Xtest)

# evaluate the score
score = accuracy_score(y_test,y_predicted)
print(score)

# find misclassified items
X_misclass = Xtest[y_test != y_predicted]

# make plot displaying the misclassified items among the other test data

# set up and select first of two subplots
plt.subplot(1,2,1)

# plot test data (sepal width vs petal width)
plt.scatter(Xtest['sepal width (cm)'],Xtest['petal width (cm)'],c=y_test)
# plot misclassified items
plt.scatter(X_misclass['sepal width (cm)'],X_misclass['petal width (cm)'],c='r')
#label axes
plt.xlabel('sepal width (cm)')
plt.ylabel('petal width (cm)')

# move to second subplot
plt.subplot(1,2,2)
# plot test data (sepal length vs petal length)
plt.scatter(Xtest['sepal length (cm)'],Xtest['petal length (cm)'],c=y_predicted)
# plot misclassified items
plt.scatter(X_misclass['sepal length (cm)'],X_misclass['petal length (cm)'],c='r')
# label axes
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')

# show plot
plt.show()
