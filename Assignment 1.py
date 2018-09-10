#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 14:05:43 2018

@author: Petra
"""

import pandas as pd
#import csv file
Data = pd.read_csv('file.csv')

#Drop the columns if there is na more than 50%
Data = Data.loc[:, Data.isnull().sum() < 0.5*Data.shape[0]]

#showing all the statisitical summary
Data.describe(include='all')

#showing total number of misssing data in columns
Data.isnull().sum()

