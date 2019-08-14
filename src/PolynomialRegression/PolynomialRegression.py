# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:36:04 2019

@author: vnandanw
"""

#import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read the data set from file
FlatData = pd.read_csv('Insurance.csv')

#Separate features and labels from the data set
X = FlatData.iloc[:,:-1].values
y = FlatData.iloc[:,1].values

#Create train and test data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Linear regression approach with train data
from sklearn.linear_model import LinearRegression
regexAgent = LinearRegression()
regexAgent.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
polyFeature = PolynomialFeatures(degree = 3)
Poly_matrix = polyFeature.fit_transform(X)

linregression2 = LinearRegression()
linregression2.fit(Poly_matrix, y)

#Plot the data on the graph
plt.scatter(X,
            y,
            color='green')
plt.plot(X, regexAgent.predict(X), color = 'red')
plt.title('Insurance Premium - Polynomial')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.show()

#Plot the data on the graph
plt.scatter(X,
            y,
            color='green')
plt.plot(X, linregression2.predict(polyFeature.fit_transform(X)), color = 'red')
plt.title('Insurance Premium - Polynomial')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.show()


val = [[68]]
predictLinear = regexAgent.predict(val)
predictPoly = linregression2.predict(polyFeature.fit_transform(val))

plt.scatter(val, predictLinear, color = 'red')
plt.plot(X, regexAgent.predict(X), color = 'blue')
plt.title('Insurance Premium - Test with Linear')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(val, predictPoly, color = 'red')
plt.plot(X, linregression2.predict(polyFeature.fit_transform(X)), color = 'green')
plt.title('Insurance Premium - Test with Polynomial')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.show()
