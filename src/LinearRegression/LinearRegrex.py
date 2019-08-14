# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:11:04 2019

@author: vnandanw
"""

#import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read the data set from file
FlatData = pd.read_csv('Price.csv')

#Separate features and labels from the data set
X = FlatData.iloc[:,:-1].values
y = FlatData.iloc[:,1].values


##Create train and test data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Linear regression approach with train data
from sklearn.linear_model import LinearRegression
regexAgent = LinearRegression()
regexAgent.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
polyFeature = PolynomialFeatures(degree = 3)
Poly_matrix = polyFeature.fit_transform(X)

regexAgent2 = LinearRegression()
regexAgent2.fit(Poly_matrix, y)

#Plot the data on the graph
plt.scatter(X,
            y,
            color='green')
plt.plot(X, regexAgent.predict(X), color = 'red')
plt.title('Compare Training result - Area/Price')
plt.xlabel('Area of Flat')
plt.ylabel('Price')
plt.show()

#Plot the data on the graph
plt.scatter(X,
            y,
            color='green')
plt.plot(X, regexAgent2.predict(polyFeature.fit_transform(X)), color = 'red')
plt.title('Compare Training result - Area/Price')
plt.xlabel('Area of Flat')
plt.ylabel('Price')
plt.show()

val = [[1500]]

linearPredict = regexAgent.predict(val)
polynomialPredict = regexAgent2.predict(polyFeature.fit_transform(val))

plt.scatter(X,
            y,
            color='green')
plt.plot(X, linearPredict, color = 'red')
plt.title('Compare Training result - Area/Price')
plt.xlabel('Area of Flat')
plt.ylabel('Price')
plt.show()

#Plot the data on the graph
plt.scatter(X,
            y,
            color='green')
plt.plot(X, polynomialPredict, color = 'red')
plt.title('Compare Training result - Area/Price')
plt.xlabel('Area of Flat')
plt.ylabel('Price')
plt.show()


