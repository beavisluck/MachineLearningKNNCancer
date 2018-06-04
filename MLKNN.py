# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 08:12:26 2018

@author: rakha
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

filename='data.csv'
df= pd.read_csv(filename)
df.replace({'diagnosis':{'M':0, 'B':1 }},inplace=True)
X=df[['radius_mean','texture_mean','area_mean','smoothness_mean']]
Y=df[['diagnosis']]

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=26)

knn.fit(X_train,y_train.values.ravel())
score=knn.score(X_test,y_test)

myvals= np.array([18,10,1001,0]).reshape(1,-1)
myvals2= np.array([13,16,520,0]).reshape(1,-1)

result = knn.predict(myvals)

print(result)

"""
malign=cancer
benign=tumor

"""
