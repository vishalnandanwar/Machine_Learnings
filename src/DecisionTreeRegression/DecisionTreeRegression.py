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


from sklearn.tree import DecisionTreeRegressor
DTRegression = DecisionTreeRegressor(random_state = 0)
DTRegression.fit(X,y)

#Plot the data on the graph
plt.scatter(X,
            y,
            color='green')
plt.plot(X, DTRegression.predict(X), color = 'red')
plt.title('Insurance Premium - Polynomial')
plt.xlabel('Age')
plt.ylabel('Premium')
plt.show()


val = [[68]]
predictLinear = DTRegression.predict(val)


