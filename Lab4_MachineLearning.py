# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:55:25 2019

@author: M7md_Karam
"""

import pandas as pd
dataset = pd.read_csv('Social_network_Ads.csv')
x = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder_z = LabelEncoder()
x[:, 0] = labelencoder_z.fit_transform(x[:,0])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/8, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier (random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix (y_test, y_pred)
accuracy_score(y_test, y_pred)