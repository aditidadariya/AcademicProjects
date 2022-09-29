#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: config.py

This file contains all the functions to be called by mainsfile.py

"""
######################## Importing necessary libraries ########################
import pandas as pd
# Import LabelEncoder to change the data types of attributes
from sklearn.preprocessing import LabelEncoder
# Import parameters.py file to get all the variables here
from config import *


########################## Function declarations ##############################

# Newline gives a new line
def Newline():
    print("\r\n")
    
# ClearLists removes all values from lists
def ClearLists():
    models.clear()
    names.clear()
    results.clear()
    basic_score.clear()
    score.clear()
    final_results.clear()
    scores_list.clear()


# ReadDataset function reads the dataset csv file and store it in a dataframe
# location_of_file is variable thats stores the file location and is defined in config.py file
def ReadDataset(location_of_file):
    # Read the data file from specified location
    df = pd.read_csv(location_of_file,header=None)
    # Specifying the column names into the dataset using dataframe, as there are no columns already specified in the dataset
    df.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
    return df


# for loop will iterate over all the columns in dataframe to find and drop the rows with special characters [4]
def RemoveSpecialChar(special_char,df):
    for eachcol in df:
        # Drop rows where "?" is displayed
        df.drop(df.loc[df[eachcol] == special_char].index, inplace=True)
    return df


# Encoder function transforms the character and object value in dataframe [5]
def Encoder(df): 
    # columnsToEncode is to get a list of columns having datatype as category and object
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    # le is an object of LabelEncoder with no parameter
    le = LabelEncoder()
    # for loop will iterate over the columns. 
    # it will first try to use LabelEncoder to fit_transform
    # and if there are any error than the except will be executed
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature) 
    return df

