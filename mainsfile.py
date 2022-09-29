#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student ID: 29837043
File: mainsfile.py

This file contains the implementation of models. 
Variables declared in config.py file will be used here
Functions declared in functionlibrary.py file will be called here to work on the algorithms

"""

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

# Load all the libraries that will be utilized through the code below
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import seaborn as sns
# Import matplotlib.pyplot to draw and save plots
import matplotlib.pyplot as plt
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import cross_val_score function
from sklearn.model_selection import cross_val_score
# Import StratifiedKFold function
from sklearn.model_selection import StratifiedKFold
# Import GridSearchCV function
from sklearn.model_selection import GridSearchCV
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
# Import confusion_matrix function
from sklearn.metrics import confusion_matrix
# Import classification_report function
from sklearn.metrics import classification_report
# Import ConfusionMatrixDisplay function to plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
# Import render from graphviz to convert dot file to png
from graphviz import render
# Import RandomUnderSampler, SMOTE, Pipeline to balance the dataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
# Import config.py file
from config import *
# Import functionlibrary.py file
from functionlibrary import *
import time


######################### STEP 2: READING THE DATA ############################


# Calling ReadDataset function from functionlibrary.py file. 
# It reads the dataset file [1] and store it in a dataframe along with its attributes
# location_of_file is defined in the config.py file
dfcredit = ReadDataset(location_of_file)

# Print the number of rows and columns present in dataset
print("The dataset has Rows {} and Columns {} ".format(dfcredit.shape[0], dfcredit.shape[1]))

# Give a new line to clearly format the output
Newline()

##################### STEP 3: PRE-PROCESSING THE DATA #########################

#======================== 3.1. Understand the data ============================

# Summarizing the data statisticaly by getting the mean, std and other values for all the columns
print(dfcredit.describe())

# From the output it is clear that the mean, std and other analysis are calculated 
# only on columns A3, A8, A11, A15, which specifies that other columns are having 
# some unnecessary data. 
# To get rid of missing values, Step 3.2 is performed

# Give a new line to clearly format the output
Newline()

#======================= 3.2. Find the missing values =========================

# Printing the missing values (NaN)
print(dfcredit.isnull().sum())

# Give a new line to clearly format the output
Newline()

# The output shows that there are no missing values (NaN) avaliable in the data

# To get a clear view of this kind of data, dataset is verified manually and found that 
# there are "?" special character specified in many columns
# Therefore Step 3.3 will find the rows having "?" and drop the respective row from the dataset

#================== 3.3. Find and drop unnecessary values =====================

# Finding the special character "?" in the dataset and droping the rows, to get a clean dataset [4]

# Calling RemoveSpecialChar function from functionlibrary.py file to remove all the special characters from dataset
dfcredit = RemoveSpecialChar(special_char,dfcredit)

# Summarizing the dataset by printing the number of rows and columns present in dataset
print("The updated dataset has Rows {} and Columns {} ".format(dfcredit.shape[0], dfcredit.shape[1]))

# Give a new line to clearly format the output
Newline()

# Summarizing the data statisticaly again by getting the mean, std and other values for all the columns
print(dfcredit.describe())

# The output again shows that mean, std etc is done on attributes A3, A8, A11 and A15 only, instead of all the attributes

# Give a new line to clearly format the output
Newline()

# Verify the data types of all attributes in the dataset, to verify the difference between the datatypes
print(dfcredit.info())

# The information shows that the datatype of attributes are of int, float and object types
# The data needs to be converted to have consistency among all the attributes, 
# because the dataset does not have a clear instruction of what all the attributes signifies to.
# Therefore, it becomes necessary to have the data of same date types

# Give a new line to clearly format the output
Newline()

#======= 3.4. Analysing and making consistent data type for all attributes ======

# Convert the column values from string/object into integer using labelencoder [5]

# In the dataset, there are mix the data types, such as values as string, float and integer
# To have same datatype, created Encoder function to convert the strings into integer values
     
# Calling the Encoder function passing the dataframe as parameter that creats a new dataframe to store the new values
dfcredit = Encoder(dfcredit)

# Save the new dataframe into a csv file
df =  pd.DataFrame(dfcredit)
df.to_csv('cleancreditdata.csv')

# Give a new line to clearly format the output
Newline()

# Summarizing the data statisticaly again by getting the mean, std and other values on all the columns
print(dfcredit.describe())

# Now the count, mean, std, min, max etc are shown for all attributes, this means the data is clean now

# Give a new line to clearly format the output
Newline()

#======================== 3.5. Class distribution =============================

# As mentioned in the dataset details, "A16" attribute is the class attribute [1]

# Finding the number of rows that belong to each class
print(dfcredit.groupby('A16').size())

# Output is showing that the class "0" is having 296 rows and class "1" is having 357 rows
# By this we understand that there are only 2 classes 0 and 1 available in the data
# which means this is a Binary Classrification type.
# This also show the 0 is assigned to denied requests for applications asking credit card approval
# and 1 is assigned for approved requests for applications asking credit card approval
# This also states that the data is imbalanced. 
# To understand the dataset more, Data Visualization by ploting the pair plot and heat graph is done in next step

# Give a new line to clearly format the output
Newline()

####################### Step 4: DATA VISUALIZATION ############################

#=========================== 4.1. Plot Pairplot ===============================

# To visualize the data, pair plot has been demonstrated below [6]

# Clear the plot dimensions
plt.clf()

# Plot the multivariate plot using pairplot
sns.set(style="ticks", color_codes=True)
grh = sns.pairplot(dfcredit,diag_kind="kde")
plt.show()

# Save the pairplot in png file
plt.savefig("pairplot.png")

# Give a new line to clearly format the output
Newline()

# Wait for the graph to be displayed
time.sleep(30)

# The plots suggests that there is a high correlation and a predictable relationship 
# between the diagonal grouping of some pairs of attributes
# As stated earlier that the data is imbalanced, the graph is not clearly linear too.

#=========================== 4.2. Plot Heat map =============================

# To visualize the data again, heat map is demonstrated below [7]

# Clear the plot dimensions
plt.clf()

# Plot Heat graph to visualise the correlation
plt.figure(figsize=(20,10))
matrix1=dfcredit.corr()
plt.title('Credit Approval Correlation', size = 15)
sns.heatmap(matrix1, vmax=10.8, square=True, cmap='YlGnBu', annot=True)

# Save the pairplot in png file
plt.savefig("heatplot.png")

# Give a new line to clearly format the output
Newline()

# Wait for the graph to be displayed
time.sleep(30)

# As observed earlier in the pairplot, the data has high correlation and a predictable relationship 
# between the diagonal grouping of some pairs of attributes. 
# The same has been observed in Hear map as well.
# To fix the imbalance problem, I have used the balancing technique RandomUnderSampler and SMOTE
# in the further steps after checking the performance of Decision Tree and Linear Discriminent Analysis models
# using the basic HoldOut Validation (i.e. train and test split)

######################## STEP 5: FEATURE SELECTION ############################

# To implement any model on the data, we need to do feature selection first
# and therefore the dataset is devided into 2 parts.
# Firstly features (all the attributes except the target attribute) and secondly target (class attribute).
# The dfcredit dataframe has been split into dfcredittarget having only column A16, as this is the class attribute,
# and all the other attributes are taken into dfcreditdata dataframe as features

dfcreditdata = dfcredit.drop("A16", axis=1) # Features
dfcredittarget = dfcredit.iloc[:,-1]    # Target

# None of the feature selection methods are used here because the dataset is having very less number of attributes.
# and therefore, the dimentionality reduction has not been performed here.
# Also, the attirbute names have been changed to meaningless by the author [1], hence all the attributes are considered as features

# $$$$$$$$$$$$$$$$$$$$$$ BUILDING DECISION TREE CLASSIFIER AND LINEAR DISCRIMINENET ANALYSIS MODELS $$$$$$$$$$$$$$$$$$$$$$$$$$$

############## STEP 6: BUILD BASE MODELS WITH HOLDOUT VALIDATION ###################

# Base Model is built below using the HoldOut Validation (train test split) to evaluate the accuracy

# ================ 6.1. Split the data to train and test set ==================

# Spot Check Algorithms with Decision Tree [8] [9] [10] and Linear Discriminant algorithms [11] [12]

# Data is splited into training and test set using the feature and target dataframes
# to understand the model performance with 70% training set and 30% test set
X_train, X_test, Y_train, Y_test = train_test_split(dfcreditdata, dfcredittarget, test_size=0.3, random_state=None)

# ========================== 6.2. Define models ===============================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier [8] [9] [10]
models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis [11] [12]
models.append(('LDA', LinearDiscriminantAnalysis()))

# ===================== 6.3. Build Model and Evaluate it  =====================

# Define BasicModel function to evaluates the models
# displays the confusion matrix and plot it [13] [14]
# displays the classification_report
def BasicModel(models):
    # Evaluate each model in turns
    for name, model in models:
        # Train the model
        modelfit = model.fit(X_train,Y_train)
        # Predict the response for test dataset
        Y_predict = modelfit.predict(X_test)
        # Store the accuracy in results
        results.append(metrics.accuracy_score(Y_test, Y_predict))
        # Store the model name in names
        names.append(name)
        # Print the prediction of test set
        print('On %s Accuracy is: %f ' % (name, metrics.accuracy_score(Y_test, Y_predict)*100))
        # Store the name and accuracy in basic_score list
        basic_score.append({"Model Name": name, "Accuracy": metrics.accuracy_score(Y_test, Y_predict)*100})
        # Print Confusion Matrix and Classification Report
        print(confusion_matrix(Y_test, Y_predict))
        print(classification_report(Y_test, Y_predict))
        # Plot Confusion Matrix [13] [14]
        cm = confusion_matrix(Y_test, Y_predict, labels=modelfit.classes_)
        cmdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelfit.classes_)
        cmdisp.plot()
        plt.show()
    return basic_score

# Calling BasicModel function
BasicModel(models)

# Create a dataframe to store accuracy
dfbasicscore = pd.DataFrame(basic_score)    
print(dfbasicscore.head())

# Give a new line to clearly format the output
Newline()

# Output:
    # On DTCLS Accuracy is: 81.632653 
    # On LDA Accuracy is: 85.204082 
# The output shows that the accuracy for LDA model is better than Decision Tree Classifier.
# To improve the performance of each model, the evaluation is done with respect to different random_state in the next step


################## STEP 7: BUILD MODELS WITH RANDOM STATE #####################

# Model is build with random state to evelaute the accuracy

# ======================= 7.1. Define models ==================================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier
models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis
models.append(('LDA', LinearDiscriminantAnalysis()))


# ===================== 7.2. Build Model and Evaluate it  =====================

# Define BuildModelRS function to evaluate the models based on the random states [8-12]
# rand_state variable has been defined in the config.py file with values 1,3,5,7
# stores the Model Name, Random State and Accuracy of each model in score list
def BuildModelRS(models,rand_state):
    # Evaluate each model in turn
    for name, model in models:
        # for loop will train and predict the decision tree model on different random states
        for n in rand_state:
            # The training set and test set has been splited using the feature and target dataframes with different random_state
            X_train, X_test, Y_train, Y_test = train_test_split(dfcreditdata, dfcredittarget, test_size=0.3, random_state=n)
            # Train Decision Tree Classifer
            modelfit = model.fit(X_train,Y_train)
            # Predict the response for test dataset
            Y_predict = modelfit.predict(X_test)
            # Store the accuracy in results
            results.append(metrics.accuracy_score(Y_test, Y_predict))
            # Store the model name in names
            names.append(name)
            # Store the Model Name, Random State and Accuracy into score list
            score.append({"Model Name": name, "Random State": int(n), "Accuracy": metrics.accuracy_score(Y_test, Y_predict)*100})
    return score

# Calling BuildModelRS to evaluate the models with random states and return the accuracy
BuildModelRS(models,rand_state)

# Create a dataframe to store accuracy
dfrsscore = pd.DataFrame(score)    
print(dfrsscore.head(8))

# Give a new line to clearly format the output
Newline()

# The output shows that the accuracy changes in each randon state. Therefore, data has been balanced in further steps.

######################## STEP 8: BALANCING THE DATA ###########################

# Installed imbalance-learn library using "conda install -c conda-forge imbalanced-learn" [15]

# To balance the data, RandomUnderSampler and SMOTE is utilized to undersample and oversample the data along with pipeline [16]

# Define oversmaple with SMOTE function
oversample = SMOTE()
# Define undersample with RandomUnderSampler function
undersample = RandomUnderSampler()
# Define Steps for oversample and undersample
steps = [('o', oversample), ('u', undersample)]
# Define the pipeline with the steps
pipeline = Pipeline(steps = steps)
# Fit the features and target using pipeline and resample them to get X and Y
X, Y = pipeline.fit_resample(dfcreditdata, dfcredittarget)
# Print the shape of X and Y
print("The features dataset has Rows {} and Columns {} ".format(X.shape[0], X.shape[1]))
print("The target dataset has Rows {} and Columns {} ".format(Y.shape[0],0))

# Give a new line to clearly format the output
Newline()

# By balancing the data using oversample and undersample, X and Y now have adequate amount of data avaialble
# Reducing the dimensionality is not needed because the dataset is small with only 15 features


######## STEP 9: OPTIMIZE MODEL ON BALANCED DATA USING CROSS VALIDATION #######

# StratifiedKFold Cross Validation is utilized to improve the performance as it splits the data approximately in the same percentage [17] [18]
# StratifiedKFold Cross Validation is used because there are 2 classes, and the split of train and test set could be properly done

# ============================ 9.1. Define models =============================

# Clear lists
ClearLists()

# Create model object of Decision Tree Classifier
models.append(('DTCLS', DecisionTreeClassifier()))
# Create model object of Linear Discriminent Analysis
models.append(('LDA', LinearDiscriminantAnalysis()))

# ===================== 9.2. Build Model and Evaluate it  =====================

# Define function BuildModelBalCV to evaluate models on the balanced data and utilize cross validation
def BuildModelBalCV(models):
    # Evaluate each model in turn  
    for name, model in models:
        # Define StratifiedKFold [17] [18]
        skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
        # Get the X and Y using StratifiedKFold
        skfold.get_n_splits(X,Y)
        # Evaluate each model with cross validation
        cv_results = cross_val_score(model, X, Y, cv=skfold, scoring='accuracy')
        # Store the accuracy in results
        results.append(cv_results)
        # Store the model name in names
        names.append(name)
        # Print the results
        print('On %s: Mean is %f and STD is %f' % (name, cv_results.mean()*100, cv_results.std()))
        # Store the Model Name, Mean and STD into score list
        score.append({"Model Name": name, "Mean": cv_results.mean()*100, "STD": cv_results.std()})
    return score
    return results
    return names

# Calling BuildModelBalCV function to get the accuracy, cross validation results and names of models used
BuildModelBalCV(models)
# Create a dataframe to store accuracy
dfscore = pd.DataFrame(score)    
print(dfscore.head())

# Give a new line to clearly format the output
Newline()

# Compare Algorithms and plot them in boxplot
pyplot.clf()
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Give a new line to clearly format the output
Newline()

# Outout:
    # On DTCLS: Mean is 82.355243 and STD is 0.033680
    # On LDA: Mean is 87.255477 and STD is 0.040058

# LDA is having the highest Mean value of 87.25%.
# Decision Tree has also performed good with Mean as 82.35%
# Looking at the boxplot and the Mean values, it is clear that LDA has performed better even after balancing the data 
# along with StratifiedKFold Cross Validation by giving Mean as 87.25%


################## STEP 10: TUNE DECISION TREE BY GRIDSEARCHCV ##################

# Tuning the Decision Tree Classifier with GridSearchCV to find the best hyperparameter [19] [20]

# Define decision tree classifier model
dtcmodel = DecisionTreeClassifier()
# Define StratifiedKFold
skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
# Define parameters
param_dict = dict()
param_dict = {"criterion": ["gini", "entropy"], "max_depth": [7,5,3]}
# Build GridSearchCV to get the accuracy
search = GridSearchCV(dtcmodel, param_dict, scoring='accuracy', cv=skfold, n_jobs=-1)
# Fit the GridSearchCV
results = search.fit(X, Y)
# Summarize 
print('Decision Tree Mean Accuracy: %f' % results.best_score_)
print('Config: %s' % results.best_params_)

# Give a new line to clearly format the output
Newline()

# Output:
    # Decision Tree Mean Accuracy: 0.868290
    # Config: {'criterion': 'entropy', 'max_depth': 3}
# The best parameter for Decision Tree Classifier are 'criterion' as 'entropy', 'max_depth' as 3, 
# using which the performance of decision tree has been increased slightly with 86.82%

###################### STEP 11: TUNE LDA BY GRIDSEARCHCV ######################

# Tuning the LDA with GridSearchCV to find the best hyperparameter [19]

# Define LDA model
ldamodel = LinearDiscriminantAnalysis()
# Define StratifiedKFold
skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
# Define parameters
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
# Build GridSearchCV to get the accuracy
search = GridSearchCV(ldamodel, grid, scoring='accuracy', cv=skfold, n_jobs=-1)
# Fit the GridSearchCV
results = search.fit(X, Y)
# Summarize
print('LDA Mean Accuracy: %f' % results.best_score_)
print('Config: %s' % results.best_params_)

# Give a new line to clearly format the output
Newline()

# Output:
    # LDA Mean Accuracy: 0.872516
    # Config: {'solver': 'svd'}

# The best parameter LDA are 'solver' as 'svd' 
# using which the performance of LDA model has been increased slightly to 87.25%


######################### STEP 12: BUILD TUNED MODEL ##########################

# Building the model finally on balanced data with best hyper-parameters along with StratifiedKFold cross validation [21]

# ========================== 12.1. Define models ==============================

# Clear lists
ClearLists()

# Define models with the best hyperparameters found earlier
models.append(('DTCLS', DecisionTreeClassifier(criterion='entropy', max_depth=3)))
models.append(('LDA', LinearDiscriminantAnalysis(solver='svd')))

# ==================== 12.2. Build Model and Evaluate it  =====================

# Define BuildFinalModel function to evaluate models with their hyper-parameters
def BuildFinalModel(models):
    # Evaluate each model in turn 
    for name, model in models:
        # Define StratifiedKFold
        skfold = StratifiedKFold(n_splits=10, random_state= None, shuffle=True)
        # Get the X and Y using StratifiedKFold
        skfold.get_n_splits(X,Y)
        # Evaluate each model with cross validation [21]
        cv_results = cross_val_score(model, X, Y, cv=skfold, scoring='accuracy')
        # Store the cross validationscore into results
        final_results.append(cv_results)
        # Store model name into names
        names.append(name)
        # Print the results
        print('On %s: Mean is %f and STD is %f' % (name, cv_results.mean(), cv_results.std()))
        # Store the Model Name, Mean and STD into score list
        score.append({"Model Name": name, "Mean": cv_results.mean(), "STD": cv_results.std()})
    return score
    return final_results

# Calling BuildFinalModel to get the accuracy after tuning
BuildFinalModel(models)
# Create a dataframe to store accuracy
dffinalscore = pd.DataFrame(score) 
print(dffinalscore.head())

# Give a new line to clearly format the output
Newline()

# Compare Algorithms and plot them in boxplot
pyplot.clf()
pyplot.boxplot(final_results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Give a new line to clearly format the output
Newline()

# Output:
    # On DTCLS: Mean is 0.859996 and STD is 0.034825
    # On LDA: Mean is 0.871264 and STD is 0.043806
    
# Looking at the Mean and Boxplot, it is clear that LDA has performed better than 
# Decision Tree after tuning the model with hyperparameters using StratifiedKFold cross validation. 
# However, there is not much difference in both the accuracies.


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$     BUILDING DEEP LEARNING MODEL     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Neural network is considered for implementing the deep learning model below [22] [23]

################ STEP 1: IMPORTING THE NECESSARY LIBRARIES ####################

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

####################### STEP 2: BUILD BASE MODEL ##############################

# ========================== 2.1. Define models ===============================

# Clear list
ClearLists()

# Define Sequential model
model = Sequential()
# Define the Flatten layer
model.add(Flatten())
# Define the hidden layer
model.add(Dense(128, input_dim=15, activation='relu'))
# Define the output layer
model.add(Dense(2, activation='softmax'))

# ==================== 2.2. Build Model and Evaluate it  ======================

# Define BasicDeepLearnModel function to evaluate the model and get the accuracy

def BasicDeepLearnModel(model):
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
    # Fit the model
    model.fit(x_train, y_train, epochs=100, batch_size=100)
    # Evaluate the model
    scores = model.evaluate(x_test, y_test)
    # Print the accuracy
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # Store the Metrics Name and Accuracy into scores_list list
    scores_list.append({"Metrics Name": model.metrics_names[1] , "Accuracy": scores[1]*100})
    return scores_list

# Calling BasicDeepLearnModel function to evaluate the deep learning model without any hyper-parameters
BasicDeepLearnModel(model)

# Create a dataframe to store accuracy
dfBasickeras = pd.DataFrame()
# Define a dataframe to with scores_list
dfBasickeras = pd.DataFrame(scores_list) 
print(dfBasickeras.head())

# Give a new line to clearly format the output
Newline()

# Output: 
    # accuracy: 79.53%
# The Accuracy of Neural Nterwork deep learning model is 79.534882. Observed that loss is between 0.3 to 3.9.


################# STEP 3: BUILD MODEL WITH VARIOUS EPOCHS #####################

# ========================== 3.1. Define models ===============================

# Clear list
ClearLists()

# Define Sequential model
model = Sequential()
# Define the Flatten layer
model.add(Flatten())
# Define the hidden layer
model.add(Dense(128, input_dim=15, activation='relu'))
# Define the output layer
model.add(Dense(2, activation='softmax'))

# ==================== 3.2. Build Model and Evaluate it  ======================

# Define BasicDeepLearnModel function to evaluate the model and get the accuracy

def BasicDeepLearnModel(model):
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # Split the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
    for n in eps:
        model.fit(x_train, y_train, epochs=n, batch_size=100)
        # Evaluate the model
        scores = model.evaluate(x_test, y_test)
        # Print the accuracy
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        # Store the Metrics Name and Accuracy into scores_list list
        scores_list.append({"Epochs": n, "Accuracy": scores[1]*100})
    return scores_list

# Calling BasicDeepLearnModel function to evaluate the deep learning model without any hyper-parameters
BasicDeepLearnModel(model)

# Create a dataframe to store accuracy
dfepochskeras = pd.DataFrame()
# Define a dataframe to with scores_list
dfepochskeras = pd.DataFrame(scores_list) 
print(dfepochskeras.head())

# Give a new line to clearly format the output
Newline()

# Find the maximum accuracy found with various epochs
df1 = dfepochskeras[["Accuracy"]].idxmax()
ind = df1.values[0]
df2 = dfepochskeras.iloc[ind:ind+1,:1]
max_epoch = df2["Epochs"].values[0]
print("The best epochs value having maximum accuracy is: ")
print (max_epoch)

# The Accuracy with the best epoch value of Neural Nterwork deep learning model is 83.720928


################# STEP 4: TUNE MODEL WITH HYPERPARAMETERS #####################

# To tune the Deep Learning Model, kerastuner has been installed using "conda install -c conda-forge keras-tuner" [24]
# kerastuner was throwing error even after installing it with above command, so installed it with "pip install keras-tuner" [25]

# ================== 4.1. Find best hyper-parameters ==========================

# Define function DLbuildmodel with the hyperparameters [26]
def DLbuildmodel(hp):
    # Define Sequential model
    model = Sequential()
    # To get the best value of number of neurons per hidden layer, hp.Int hyperparameter is used 
    # by giving a range from min_value as 32 and max_value as 300 with the step size as 50
    model.add(Dense(units= hp.Int('units', min_value=32, max_value=300, step=50), activation='relu'))
    # Define the output layer
    model.add(Dense(2, activation='softmax'))
    # compile model - To get the best Adam optimizer, learning_rate from 0.01 and 0.001 is used
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-4])),metrics=['accuracy'])
    return model
    
# ================ 4.2. Tune model with RandomSearch ==========================

# To create an instance of RandomSearch class, RandomSearch function is called
# Calling the model building funtion "DLbuildmodel"
# To maximize the accuracy "objective" is set to "val_accuracy"
# max_trails is set to 5 to limit the number of model variations to test
# executions_per_trial is set to 3 to limit the number of trials per variation

tuner = RandomSearch(DLbuildmodel, objective = 'val_accuracy', max_trials=5, executions_per_trial=3, directory = 'my_dir', project_name='deeplearningmodel')

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
# Train the model
tuner.search(x_train,y_train,epochs=max_epoch,validation_data=(x_test,y_test))
# Retrieve the best_model
best_model = tuner.get_best_models(num_models=1)[0]
# Get the loss and mse from the best model
loss, mse = best_model.evaluate(x_test, y_test)

# Give a new line to clearly format the output
Newline()

# Summarize the results with best hyper-parameters
tuner.search_space_summary()

# Give a new line to clearly format the output
Newline()

# Print the best hyperparameter
print(tuner.get_best_hyperparameters()[0].values)

# Give a new line to clearly format the output
Newline()

# Output:
    # Best val_accuracy So Far: 0.8790697654088339
    # Total elapsed time: 00h 06m 25s
    # INFO:tensorflow:Oracle triggered exit
    # 7/7 [==============================] - 0s 1ms/step - loss: 0.4322 - accuracy: 0.8599


    # Search space summary
    # Default search space size: 2
    # units (Int)
    # {'default': None, 'conditions': [], 'min_value': 32, 'max_value': 300, 'step': 50, 'sampling': None}
    # learning_rate (Choice)
    # {'default': 0.01, 'conditions': [], 'values': [0.01, 0.0001], 'ordered': True}


    # {'units': 132, 'learning_rate': 0.01}

####################### STEP 5: Final Deep Learning Model #####################

# Define list
scoreslist = []
# Clear list
scoreslist.clear()

# Define model
model = Sequential()
# Define the layers with best units value
model.add(Dense(132, input_dim=15, activation='relu'))
model.add(Dense(2, activation='softmax'))

# compile model with best learning_rate
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# Split the dataset to train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)

# fit the model with best epoch value
modelfit = model.fit(x_train, y_train, epochs=max_epoch, batch_size=100)

# evaluate the model
scores = model.evaluate(x_test, y_test)

# print the accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scoreslist.append({"Metrics Name": model.metrics_names[1] , "Accuracy": scores[1]*100})

# Create a dataframe to store accuracy
dfkeras = pd.DataFrame()
#scoreslist.append({"Metrics Name": model.metrics_names[1] , "Accuracy": scores[1]*100})
dfkeras = pd.DataFrame(scoreslist) 
print(dfkeras.head())

# Output:
    # Observed that the accuracy varies between 80% to 94% after applying the best epoch value and best hyperparameter 
    # It is seen that the last trained model with epoch as 300 gives accuracy as 93.46%, however the best accuracy is 86.05%
    # Observed that loss is between 0.19 to 0.25


#################### STEP 6: PLOT LOSS AND ACCURACY ###########################

# Plotting Loss and Accuracy [27]

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(modelfit.history['loss'], label='X_train')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(modelfit.history['accuracy'], label='X_test')
pyplot.legend()
pyplot.show()

# By the loss and accuracy plotted below, it is observed that the accuracy is increased by the decrease in loss.

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$     CONCLUSION OF MODELS BUILT     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Decision Tree model produced an accuracy of 81.63% on single modelling training
# Linear Discriminant Classifier model produced an accuracy of 85.20% on single modelling training
# The Decision Tree and Linear Discriminant Classifier models performed good after balancing the data and 
# applying few hyperparameter along with cross validation method. 
# Performance of Decision Tree model increased to 85.99%, which is an increase of 4.36% after tuning the model
# whereas LDA model performed slightly better than Decision Tree by giving accuracy of 87.12%,
# which is an increase of just 1.92% after tuning the model.
# There wasn't much difference between both models after they were tuned.
# The Neural Network Deep Learning model produced an accuracy of 79.53% on single modelling training. 
# The performance of Neural Network is good with 85.05% after tuning the model with hyperparameters, 
# which is a significant increase of 6.52% after the model was tuned.
# Also, the loss has been drastically changed from (0.3 to 3.9) to (0.19 to 0.25).
# Considering only the accuracy of the three model, LDA has the highest accuracy. 
# however Neural Network performed much better than Decision Tree and LDA models after it was tuned.
# Hence Neural Network Deep Learning model is recommended to be utilized to prove the approval of credit cards.


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ REFERNCES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# [1] https://archive.ics.uci.edu/ml/datasets/Credit+Approval
# [4] https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
# [5] https://www.projectpro.io/recipes/convert-string-categorical-variables-into-numerical-variables-using-label-encoder
# [6] https://pythonbasics.org/seaborn-pairplot/
# [7] https://seaborn.pydata.org/generated/seaborn.heatmap.html
# [8] https://scikit-learn.org/stable/modules/tree.html
# [9] https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# [10] https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# [11] https://machinelearningmastery.com/linear-discriminant-analysis-with-python/
# [12] https://www.statology.org/linear-discriminant-analysis-in-python/
# [13] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# [14] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# [15] https://anaconda.org/conda-forge/imbalanced-learn
# [16] https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
# [17] https://www.kaggle.com/vitalflux/k-fold-cross-validation-example/
# [18] https://scikit-learn.org/stable/modules/cross_validation.html
# [19] https://machinelearninghd.com/gridsearchcv-classification-hyper-parameter-tuning/
# [20] https://ai.plainenglish.io/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda
# [21] https://www.kaggle.com/vitalflux/k-fold-cross-validation-example/
# [22] https://keras.io/guides/training_with_built_in_methods/
# [23] https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# [24] https://anaconda.org/conda-forge/keras-tuner
# [25] https://keras.io/api/keras_tuner/tuners/
# [26] https://keras.io/api/keras_tuner/hyperparameters/#hyperparameters-class
# [27] https://pyimagesearch.com/2021/06/07/easy-hyperparameter-tuning-with-keras-tuner-and-tensorflow/