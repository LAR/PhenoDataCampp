# DataCampp Machine Learning for image data, Pandas examples

import pandas as pd


# Example 1 - manually created dataframe

list1 = [1.3, 1.4, 0.8, 2.6]
list2 = ['daisy', 'daisy', 'clover', 'daffodil']

df = pd.DataFrame(data = {'size':list1, 'species':list2})
print(df)
print()
print(df.dtypes)

# Example 2 - CSV imported data

import pandas as pd
imp_df = pd.read_csv('my_data.csv')
print()
print('imported data') 
print(imp_df)

# Example 3  - importing the Iris data

from sklearn.datasets import load_iris
iris = load_iris()

# Get the data and convert to DataFrame - its stored in iris.data
iris_df = pd.DataFrame(iris.data) 

# Set the column names - these are stored as iris.feature_names
iris_df.columns = iris.feature_names

print(iris_df.head())


# Example 4 - important as DataFrame

iris = load_iris(as_frame=True)

alt_df = iris.frame
print(alt_df.head())


X = alt_df.drop('target', axis=1) # axis=1 is essential to drop columns rather than rows
print(X.head())

y = alt_df['target']
