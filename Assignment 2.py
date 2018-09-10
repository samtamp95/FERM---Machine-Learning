#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:07:10 2018

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

#create the number of columns(variables) that will be modeled
c=list(range(len(X[0])))

# Encoding categorical data
#Enconding the Independant Variable
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder = LabelEncoder()

#X = Data.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#removing one dummy to reduce rendundancy
#X = X[:, 1:]                #index one to the end

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#p values
#ones. to change into ones
#X = np.append(arr = np.ones((len(Data),1)).astype(int), values = X, axis = 1)

#Elimination by P-Value and adjusted r-square
#this is all done by 

#setting why value for the backward elimination
y=y_train

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    #determine temp figure out the dataframe size then run
    temp = np.zeros((len(X_opt),len(X_opt[0]))).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
#setting the significane level to 5%
SL = 0.05

X_opt = X_train[:, c]

#optimized X values that are needed
X_Modeled = backwardElimination(X_opt, SL)



#testing to predict the value from the model that has been tested
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Modeled, y_test)         #fit regressor to training set

# Predicting the Test set results
y_pred = regressor.predict(X_Modeled)
y_pred=pd.DataFrame(y_pred)
y_pred.columns = ['Predicted']
#y_pred.to_csv('predicted.csv')



























'''

# optimization
import numpy as np
from scipy.optimize import least_squares as lsq
from scipy.optimize import minimize
# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html


# define Matrix X
X0 = pd.DataFrame(np.ones(len(Data)),columns=['X0'])
X = pd.concat((X0,Data.iloc[:,Data.columns != 'oibdp']),axis=1) 

#Y values must be in list if used in DataFrame does not work
Y=Data['oibdp'].tolist()


# error function to be minimized
def error_min (Beta):
    Y_hat = np.dot(X,Beta)
    error_hat = Y - Y_hat
    return np.dot(error_hat.transpose(),error_hat) # to use with pandas, indices should have the same name

c=list(np.repeat(100, len(Data.columns) ))

least_sq = lsq(error_min , c)
minimize = minimize(error_min , c,method='BFGS')

print(least_sq.x)
print(minimize.x)


# Matrix Form (Theoretical Approach)
a1 = np.dot(X.transpose(),X)
a2 = np.linalg.inv(a1)
a3 = np.dot(a2,X.transpose())
a4 = np.dot(a3,Y)


# Using SkLearn
from sklearn import linear_model

X1 = X.drop(X0, axis = 1)

regr = linear_model.LinearRegression()
sk_reg = regr.fit(X1,Y)
sk_reg.coef_
sk_reg.intercept_


# write a predict function that gets a model and X dataset, and predicts values

# Using skLearn model
def pred_sk (model, X):
    beta = np.insert(model.coef_,0,model.intercept_)
    return np.dot(X,beta)

print(pred_sk(sk_reg,X))

# Using our model 
def pred (model, X):
    return np.dot(X,model.x)

print(pred(minimize,X))

print(sk_reg.predict(X1))




# write an r-squared function using the pred method, get Y and 

from sklearn.metrics import r2_score
def r_sq():
    residual = Y - pred(minimize,X)
    SS_res = np.dot(residual.transpose(),residual)
    
    mean_Y = np.mean(Y)
    mean_diff = Y - mean_Y
    SS_tot = np.dot(mean_diff.transpose(),mean_diff)
    return (1 - SS_res/SS_tot)

print(r_sq())
print(r2_score(Y,pred(minimize,X)))




# define the variance-covariance matrix

residual = residual = Y - pred(minimize,X)
var_res = np.dot(residual.transpose(),residual) / (X.shape[0]-X.shape[1])
inverse_X_prime_X = np.linalg.inv(np.dot(X.transpose(),X))
var_cov = var_res * np.dot(inverse_X_prime_X,np.eye(X.shape[1]))
coeff_std = np.sqrt(np.diag(var_cov))
print(coeff_std)

coeff_t = minimize.x / coeff_std
print(coeff_t)

print(2 - 2*(scipy.stats.norm.cdf(np.abs(coeff_t))))




import statsmodels.api as sm
ols = sm.OLS(Y, X)
ols_result = ols.fit()

print(minimize.x)
print(ols_result.summary())
'''
