# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:51:09 2021

@author: giris
"""
import pandas as pd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization

df=pd.read_csv("C:\\Users\\giris\\Desktop\\kaggle\\ML-1\\adult.csv")
print(df)
print(df.shape)# view dimensions of dataset
print(df.head())
print(df.info())# view summary of dataset
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names

df.columns

#segregate the dataset into categorical and numerical variables. 
#There are a mixture of categorical and numerical variables in the dataset.
#Categorical variables have data type object.
# Numerical variables have data type int64.

categorical = [var for var in df.columns if df[var].dtype=='O']# find categorical variables

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)
print(df[categorical].head())

#There are 9 categorical variables.
#The categorical variables are given by workclass, education, marital_status, occupation, relationship, race, sex, native_country and income.
#income is the target variable

print(df[categorical].isnull().sum())#Missing values in categorical variables
#We can see that there are no missing values in the categorical variables.
## view frequency counts of values in categorical variables
for var in categorical: 
    
   print(df[var].value_counts())
#view frequency distribution of categorical variables
for var in categorical: 
        print(df[var].value_counts()/np.float64(len(df)))
#Now, we can see that there are several variables like workclass, occupation and native_country which contain missing values
#explore these variables and replace ? with NaN.

print(df.workclass.unique())# check labels in workclass variable
df.workclass.value_counts()# check frequency distribution of values in workclass variable
#We can see that there are 1836 values encoded as ? in workclass variable. I will replace these ? with NaN.
df['workclass'].replace('?', np.NaN, inplace=True)#replace '?' values in workclass variable with `NaN`
print(df.workclass.value_counts())# again check the frequency distribution of values in workclass variable
# missing value check for occupation column.
print(df.occupation.unique())#check the labels of occupation
print(df.occupation.value_counts())# check frequency distribution of values in occupation variable
df['occupation'].replace('?', np.NaN, inplace=True)
print(df.occupation.value_counts())# again check the frequency distribution of values in occupation variable
print(df.native_country.unique())# check labels in native_country variable
print(df.native_country.value_counts())# check frequency distribution of values in native_country variable
df['native_country'].replace('?', np.NaN, inplace=True)
print(df.native_country.value_counts())
print(df[categorical].isnull().sum())#Check missing values in categorical variables again
#Now, we can see that workclass, occupation and native_country variable contains missing values.

#A high number of labels within a variable is known as high cardinality.
#High cardinality may pose some serious problems in the machine learning model.
# check for cardinality in categorical variables
for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
# find numerical variables


numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

print(df[numerical].head())
#Missing values in numerical variables
print(df[numerical].isnull().sum())#We can see that all the 6 numerical variables do not contain missing values.

X = df.drop(['income'], axis=1)
y = df['income']

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print(X_train.shape, X_test.shape)
#Feature Engineering is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.

# check data types in X_train

print(X_train.dtypes)

# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

print(categorical)

# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

print(numerical)

# print percentage of missing values in the categorical variables in training set

print(X_train[categorical].isnull().mean())

# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))

# impute missing categorical variables with most frequent value
df2= pd.DataFrame(X_train,X_test)
print(df2)
#for df2 in [X_train, X_test]:
df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)    
# check missing values in categorical variables in X_train
print(X_train[categorical].isnull().sum())
# check missing values in categorical variables in X_test

print(X_test[categorical].isnull().sum())

# check missing values in X_train

print(X_train.isnull().sum())

# check missing values in X_test

print(X_test.isnull().sum())#We can see that there are no missing values in X_train and X_test

#encode the categorical value
# print categorical variables

print(categorical)
print(X_train[categorical].head())


import category_encoders as ce
 #encode remaining variables with one-hot encoding

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)#print(X_train.head())
print(X_train.shape)
cols = X_train.columns
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.head())
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
# compare the train-set and test-set accuracy to check for overfitting.
y_pred_train = gnb.predict(X_train)
print(y_pred_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))