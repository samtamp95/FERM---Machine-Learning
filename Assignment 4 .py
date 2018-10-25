#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:17:02 2018

@author: Petra
"""

import pandas as pd
#get cds data
df = pd.read_stata('cds_spread5y_2001_2016.dta')

#separating the dates
df['Date'] = pd.to_datetime(df['mdate'])

#separating the months
df['Month'] = df['Date'].dt.month

#seperating the years
df['Year'] = df['Date'].dt.year


# Set a default value
df['Quarter'] = '4'
#Quarter if above 9 months will be at quarter 4
df['Quarter'][df['Month'] > 9] = '4'
#setting for quarter 3
df['Quarter'][(df['Month'] > 6) & (df['Month'] < 9)] = '3'
#setting for quater 2
df['Quarter'][(df['Month'] > 3) & (df['Month'] < 6)] = '2'
#setting for quater 1
df['Quarter'][df['Month'] < 3] = '1'

#changing columns to float so that it can be connected to
df['gvkey'] = df['gvkey'].astype(float)

df['Quarter'] = df['Quarter'].astype(float)

df['Year'] = df['Year'].astype(float)


#get company data
Company = pd.read_csv("Quarterly Merged CRSP-Compustat.csv")

#changing columns to something similar
Company=Company.rename(columns = {'datadate':'mdate'})
Company=Company.rename(columns = {'GVKEY':'gvkey'})



#separating the dates
Company['Date'] = pd.to_datetime(Company['mdate'])

#separating the months
Company['Month'] = Company['Date'].dt.month

#seperating the years
Company['Year'] = Company['Date'].dt.year



# Set a default value
Company['Quarter'] = '4'
#Quarter if above 9 months will be at quarter 4
Company['Quarter'][Company['Month'] > 9] = '4'
#setting for quarter 3
Company['Quarter'][(Company['Month'] > 6) & (Company['Month'] < 9)] = '3'
#setting for quater 2
Company['Quarter'][(Company['Month'] > 3) & (Company['Month'] < 6)] = '2'
#setting for quater 1
Company['Quarter'][Company['Month'] < 3] = '1'

#changing columns to float so that it can be connected to
Company['gvkey'] = Company['gvkey'].astype(float)

Company['Quarter'] = Company['Quarter'].astype(float)

Company['Year'] = Company['Year'].astype(float)



#merging the data 
Data_Take=pd.merge(Company, df, on=['gvkey', 'Quarter','Year'])
