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
healtData = pd.read_csv('Health.csv')

#Separate data features and labels
X = healtData.iloc[:,:-1].values


z = healtData.iloc[:,[3,4]].values
y = healtData.iloc[:,4].values

from sklearn.impute import SimpleImputer as Imputer

missingValueImputer = Imputer(missing_values=np.nan,
                              strategy="")

missingValueImputer = missingValueImputer.fit(X[:,[3,4]])

X[:,[3,4]] = missingValueImputer.transform(X[:,[3,4]])
