#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:01:35 2020

@author: ashish
"""

#import numpy as np
#from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table("fruit_data_with_color.txt")

fruit_name = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))

print(fruit_name)

X = fruits[['mass','width','height']]
Y = fruits['fruit_label']
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,random_state=0)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, Y_train)

print(knn.score(X_test,Y_test))

print("Enter mass, width, height")
mass, width, height = map(float,input().split())
fruit_predict = knn.predict([[mass, width, height]])
print(fruit_name[fruit_predict[0]])

''' plotting for data visualization 

cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train, c = Y_train, marker = 'o', s=40,hist_kwds={'bins':15} ,figsize=(12,12),cmap=cmap)
'''