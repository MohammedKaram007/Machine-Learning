# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:09:12 2019

@author: M7md_Karam
"""

import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc [: , :-1].values
y = dataset.iloc[:, 1].values
#iloc separates values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3.0, random_state = 0)
	
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
