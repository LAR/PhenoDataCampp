

## Getting the data (Method 1)

"""
The first thing we need to do is load the Iris dataset. Scikit learn provides a
helper function called 'load_iris' we can access as follows:
"""


from sklearn.datasets import load_iris

iris = load_iris()

"""
The function load_iris returns a scikit learn 'bunch'
object. You can think of this as a Python dictionary where you can access the contents
as attributes as well as keys.

You can view the names of these attributes/keys in same way you would a regular Python
dictionary:

"""
print(iris.keys())
"""
As you might expect 'DESCR' and 'filename' contain a description of the dataset
and the full path the the data file itself (feel free to have a look at these yourself)

The data itself is conveniently already split into the format we want, i.e. a features
matrix and target vector, contained in 'data' and 'target' respectively. Similarly, the feature names
and target names are contained in the attributes 'feature_names' and 'target_names'.

"""
print(iris.feature_names,iris.data.shape)
print(iris.target_names,iris.target.shape)
"""
As you can see if you run the above code, the dataset contains 150 observations
with 4 features (width and length in cm for both petals and sepals of the flowers).
The target vector has length 150, and contains the species of each plant, which
consists of a list of integers (0,1 and 2 in this case) that refer to the list of target names
['setosa', 'versicolor','virginica']. For example, if target = 0 it is species 'setosa',
target = 1 then the species is 'versicolor', and so on.


Finally, we can use the shorthand of X and y for the variable names of our features matrix and target vector:
"""
X = iris.data
y = iris.target


"""
Now we have the data in the right format we can use it to train a machine learning model.
Before we do that though its probably worth having a closer look at the data itself.
We can do this using Matplotlib by making scatter plots of different features of the model against each other
and colouring with the species. For example comparing length and width:
"""
import matplotlib.pyplot as plt



plt.subplot(1,2,1)
plt.scatter(X[:,1],X[:,3],c=y)
plt.xlabel('sepal width (cm)')
plt.ylabel('petal width (cm)')
plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,2],c=y)
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.show()

"""
Clearly, even just looking at these two pairs of features, we can see how the three coloured species
fall into distinct clusters (though there is some overlap). The aim of machine learning in this case is to take this data,
and use it to make a model that will predict (with some accuracy hopefully!) the species
of new data-points not previously seen by the model.

A simple model we can try is the K-nearest neighbour (KNN) method (see the article/video for details
of the method itself, and what the parameter 'n_neighbours' means). To import and train the model its as simple as the following code:
"""
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

"""
Then we can use the model to make predictions of the species for the existing data:
"""
y_predicted = model.predict(X)
"""
and compare with the true values in the target vector using scikit learns accurcy metric:
"""
from sklearn.metrics import accuracy_score
score = accuracy_score(y,y_predicted)
print(score)
"""
So our newly trained model is able to make predictions on the data with 100% success.
Pretty good right? Not so fast!
We've fallen into the classic trap here of using the same data to train the model
as well as evaluate it. Doing this will generally lead to overfitting your models,
especially in the case of KNN with the number of neighbours set to one (more on overfitting later in the course).

Instead, its much better practice to split your datasets into training and testing subsets,
and use the first set to train the model, and the second to test it.
Scikit learn has a convenient function to do just this called train_test_split,
which randomly splits the data into training and test sets (default ratio is 75% training to 25% test examples)
"""
from sklearn.model_selection import train_test_split
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, random_state=23)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(Xtrain,y_train)

y_predicted = model.predict(Xtest)
score = accuracy_score(y_test,y_predicted)
print(score)

"""
Now you should see a lower accuracy score, but the model is much more likely to generalise to new, unseen data.

Lets see which of the test instances have been misclasified:

"""
X_misclass = Xtest[y_test != y_predicted]



plt.subplot(1,2,1)
plt.scatter(Xtest[:,1],Xtest[:,3],c=y_test)
plt.xlabel('sepal width (cm)')
plt.ylabel('petal width (cm)')

plt.scatter(X_misclass[:,1],X_misclass[:,3],c='r')
plt.xlabel('sepal width (cm)')
plt.ylabel('petal width (cm)')

plt.subplot(1,2,2)
plt.scatter(Xtest[:,0],Xtest[:,2],c=y_predicted)
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')

plt.scatter(X_misclass[:,0],X_misclass[:,2],c='r')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')

plt.show()


"""
The true values are plotted as before, with the misclassified values highlighted in red.
"""
