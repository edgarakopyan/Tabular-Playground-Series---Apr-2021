import pandas as pd
import os
import numpy as np
import sklearn.preprocessing as skp

# Check if directory is correct
os.getcwd()

# Upload our data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Get indeces of test values and delete Passenger ID columns in both datasets
Test_IDs = test_df['PassengerId']
test_df = test_df.drop(['PassengerId'], axis = 1)
train_df = train_df.drop(['PassengerId'], axis = 1)

# Delete names column
test_df = test_df.drop(['Name'], axis = 1)
train_df = train_df.drop(['Name'], axis = 1)

# encode sex column. Before we do that, we check that there are no missing values. We then drop those columns

test_df[test_df['Sex'].isna()]
train_df[train_df['Sex'].isna()]

train_df = pd.concat([train_df,pd.get_dummies(train_df['Sex'])], axis = 1)
test_df = pd.concat([test_df, pd.get_dummies(train_df['Sex'])], axis = 1)

test_df = test_df.drop([ 'Sex','male'], axis = 1)
train_df = train_df.drop(['Sex','male'], axis = 1)

# Create a new column with cabin letter only and one with cabin number only

train_df['Cabin_Letter'] = ''
train_df['Cabin_Number'] = ''

for i in train_df[train_df['Cabin'].notna()].index:
        train_df['Cabin_Letter'][i] = train_df['Cabin'][i][0]
        train_df['Cabin_Number'][i] = train_df['Cabin'][i][1:]

for i in train_df[train_df['Cabin'].isna()].index:
        train_df['Cabin_Letter'][i] = np.nan
        train_df['Cabin_Number'][i] = np.nan


