# -*- coding: utf-8 -*-
"""
Created on Wed May  1 06:27:08 2019

@author: vnandanw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#open the data file and import all the data in obejcts
SCM_Data = pd.read_csv("C:\Workspace\Extra_Activity\Machine_Learnings\DataSets\Smart Cooler Management\CoolerHealthData-4-29-2019-10_52-AM.csv")

#Separate data features and labels
X = SCM_Data.iloc[:,:].values
z =  SCM_Data.iloc[:,3:4].values

from sklearn.impute import SimpleImputer as Imputer

missingValueImputer = Imputer(missing_values=np.nan,
                              strategy="mean")

missingValueImputer = missingValueImputer.fit(X[:,[3,4]])

X[:,[3,4]] = missingValueImputer.transform(X[:,[3,4]])

#import csv

#csv.register_dialect('myDialect', quoting=csv.QUOTE_ALL,skipinitialspace=True)
#with open('C:\Workspace\Extra_Activity\Machine_Learnings\DataSets\Smart Cooler Management\CoolerHealthData_modified.csv', 'w') as f:
#    writer = csv.writer(f, dialect='myDialect')
#    for row in X:
#        writer.writerow(row)

#f.close()

from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()

