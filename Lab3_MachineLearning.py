# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:31:09 2019

@author: M7md_Karam
"""

import pandas as pd
dataset = pd.read_csv('Social_network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, [4]].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix (y_test, y_pred)