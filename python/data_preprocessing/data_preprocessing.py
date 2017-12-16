# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:14:05 2017

@author: saurav
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Extracting independent variables, first 3 columns
# dataset.iloc[All Rows, All columns except last(fourth) one]
X = dataset.iloc[:, :-1].values

# Extracting dependent variables, last(fourht) column
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
# imputer = Imputer(missing_values="NaN", strategy="median", axis=0)
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
"""
Replacing missing values of specific columns
imputer = imputer.fit(X[:, 1:2])
X[:, 1:2] = imputer.transform(X[:, 1:2])
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
"""

# Replacing missing values for series of columns
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encodeCountry = LabelEncoder()
X[:, 0] = encodeCountry.fit_transform(X[:, 0])
"""
      [[0, 44.0, 72000.0],
       [2, 27.0, 48000.0],
       [1, 30.0, 54000.0],
       [2, 38.0, 61000.0],
       [1, 40.0, 63777.77777777778],
       [0, 35.0, 58000.0],
       [2, 38.77777777777778, 52000.0],
       [0, 48.0, 79000.0],
       [1, 50.0, 83000.0],
       [0, 37.0, 67000.0]],
      
      Above data encodes country to 0, 1, and 2. These numbers can represent
      order(0 < 1 < 2). In our dataset all countries are equal and should not
      be ordered after encoding.
"""

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
"""
now country is encoded with dummy variables
1	0	0	44	72000
0	0	1	27	48000
0	1	0	30	54000
0	0	1	38	61000
0	1	0	40	63777.8
1	0	0	35	58000
0	0	1	38.7778	52000
1	0	0	48	79000
0	1	0	50	83000
1	0	0	37	67000

"""

"""
Ml Models will understand that independent variables are categorical
variables so we don't need to use dummy variables to encode y.
"""
encodePurchased = LabelEncoder()
y = encodePurchased.fit_transform(y)

#Splitting the Data Set into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                            random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# As fit is already applied to training set so we don't need to fit test set
X_test = sc_X.transform(X_test)

"""
Scaling only particular columns
X_train[:, 3:5] = sc_X.fit_transform(X_train[:, 3:5])
# As fit is already applied to training set so we don't need to fit test set
X_test[:, 3:5] = sc_X.transform(X_test[:, 3:5])
"""