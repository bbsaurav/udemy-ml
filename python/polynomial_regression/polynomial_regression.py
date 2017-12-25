# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:18:08 2017

@author: saurav
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
"""
Column1 Position is not important as we have different Levels for each Postion
Also we use  as matrix so we are using
X = dataset.iloc[:, 1:2].values
and not X = dataset.iloc[:, 1].values because it makes vector of int64 and not
matrix
"""
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""
As dataset is very small and unique so there is no need to train our model
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
#poly_reg = PolynomialFeatures(degree = 2)
"""
Transform X to X_poly
y = b1*x will be transferred to
y = b0*0 + b1*x + b2*x2
"""
X_poly = poly_reg.fit_transform(X)
# Fitting Linear Regression to X_poly
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Visualising the Linear Regrssion Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Salary Vs Position Level (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Results
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_poly.predict(poly_reg.fit_transform(X)), color = 'blue')
#plt.plot(X_grid, lin_reg_poly.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Salary Vs Position Level (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Multiple Regression
lin_reg_poly.predict(poly_reg.fit_transform(6.5))