# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#first thing
#set PWD by saving the file and running it
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#import data set
healtData = pd.read_csv('C:\Workspace\Extra_Activity\Machine_Learning_Activity\Health.csv')

#Separate data features and labels
X = healtData.iloc[:,:-1].values


z = healtData.iloc[:,[3,4]].values
y = healtData.iloc[:,4].values

from sklearn.impute import SimpleImputer as Imputer

missingValueImputer = Imputer(missing_values=np.nan,
                              strategy="mean")

missingValueImputer = missingValueImputer.fit(X[:,[3,4]])

X[:,[3,4]] = missingValueImputer.transform(X[:,[3,4]])

#Delaing with categorical data
#Encode data -- labelencoder
#Remove mathematical weightage  -- OnehotEncoder class

from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()

X[:,0] = X_labelencoder.fit_transform(X[:,0])
X[:,1] = X_labelencoder.fit_transform(X[:,1])
X[:,2] = X_labelencoder.fit_transform(X[:,2])
X[:,5] = X_labelencoder.fit_transform(X[:,5])

#X_labelencoder.classes_

#Use sklearn.preprocessing.oneHotEnoder to remove mathematical weightage 


#Step6 Creating training data and testing data splits
#Training data 80%
#testing data 20%

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

#Step7 Feature scalling
from sklearn.preprocessing import StandardScaler #Normal distribution
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
