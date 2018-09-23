#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:23:11 2018

@author: Petra
"""

#OIBDP is using as operating profit
#if want to calculate opearting profit revenue-COGS
#using OIBDP= operating income before depreciation 


import pandas as pd
#import csv file
Data = pd.read_csv('compustat_annual_2000_2017_with link information.csv')

#Drop the columns if there is na more than 70%
Data = Data.loc[:, Data.isnull().sum() < 0.7*Data.shape[0]]

#showing only number that will be used
Data=Data.select_dtypes(include=['float64'])


#finding columns that are not within the 1% and 99% quantile
for col in Data.columns:
    percentiles = Data[col].quantile([0.01,0.99]).values
    Data[col][Data[col] <= percentiles[0]] = percentiles[0]
    Data[col][Data[col] >= percentiles[1]] = percentiles[1]
#dropping the columns that are not within those quantile
Data=Data[Data.columns[~Data.columns.str.contains(col)]]    

#filling all data with median for each category
Data=Data.fillna(Data.median())


# Multiple Linear Regression

# Importing the libraries
import numpy as np


#only get take the ones needed to do calculation
X = Data.iloc[:, Data.columns != 'oibdp'].values
#choose the dependent variable or resulting value
y = Data['oibdp'].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0, max_depth=4,min_samples_leaf=30)
regressor.fit(X_train, y_train)

y_pred_4d_depth=regressor.predict(X_test)
y_pred_4d_depth



regressor_2 = DecisionTreeRegressor(random_state = 0, max_depth=5,min_samples_leaf=30)
regressor_2.fit(X_train, y_train)

y_pred_5_depth=regressor_2.predict(X_test)
y_pred_5_depth








