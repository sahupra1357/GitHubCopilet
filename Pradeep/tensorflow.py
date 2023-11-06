build tensorflow network for regression
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# load the dataset
boston = datasets.load_boston()
print(boston.keys())
print(boston.data.shape)
print(boston.feature_names)

# convert the dataset into a dataframe
bos = pd.DataFrame(boston.data)
print(bos.head())
bos.columns = boston.feature_names
print(bos.head())
print(boston.target.shape)
bos['PRICE'] = boston.target
print(bos.head())

# check for missing values
print(bos.isnull().sum())

# split the dataset into train and test
X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 3)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(13, activation='relu'))
model.add(tf.keras.layers.Dense(13, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=100, verbose=0)
Y_pred = model.predict(X_test)
print(Y_pred)
print(Y_test)

# check the accuracy of the model
print(np.sqrt(mean_squared_error(Y_test, Y_pred)))
print(r2_score(Y_test, Y_pred))

# plot the graph
plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()

