# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:39:41 2017

@author: saurav
"""
# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""
As dataset is very small and unique so there is no need to train our model
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
#regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting the new result
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression Result
# X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Salary Vs Position Level (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()