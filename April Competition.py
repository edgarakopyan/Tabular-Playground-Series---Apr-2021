import pandas as pd
import os
import numpy as np
import sklearn.preprocessing as skp
import sklearn.linear_model
import sklearn.ensemble

pd.options.mode.chained_assignment = None

# Check if directory is correct
os.getcwd()

# Upload our data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Get indeces of test values and delete Passenger ID columns in both datasets
Test_IDs = test_df['PassengerId']
test_df = test_df.drop(['PassengerId'], axis = 1)

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


test_df['Cabin_Letter'] = ''
test_df['Cabin_Number'] = ''

for i in test_df[test_df['Cabin'].notna()].index:
        test_df['Cabin_Letter'][i] = test_df['Cabin'][i][0]
        test_df['Cabin_Number'][i] = test_df['Cabin'][i][1:]

for i in test_df[test_df['Cabin'].isna()].index:
        test_df['Cabin_Letter'][i] = np.nan
        test_df['Cabin_Number'][i] = np.nan

# Try a rudimentary logistic model
model = sklearn.linear_model.LogisticRegression().fit(train_df[['SibSp','Pclass','Parch', 'female']], train_df.Survived) # 'Pclass',
prediction1_df = pd.Series(model.predict(test_df[['SibSp','Pclass','Parch', 'female']]))
prediction1_df = pd.concat([Test_IDs, prediction1_df], axis = 1)
prediction1_df = pd.DataFrame(prediction1_df)
prediction1_df.columns = ['PassengerId', 'Survived']
prediction1_df.to_csv('prediction1.csv', index = False)

# Try rudimentary Random Forest
forest1 = sklearn.ensemble.RandomForestClassifier(n_estimators = 125).fit(train_df[['SibSp','Pclass','Parch', 'female']],
                                                                          train_df.Survived)
prediction2_df = forest1.predict(test_df[['SibSp','Pclass','Parch', 'female']])
prediction2_df = pd.DataFrame(prediction2_df)
prediction2_df = pd.concat([Test_IDs, prediction2_df], axis = 1)
prediction2_df.columns = ['PassengerId', 'Survived']
prediction2_df.to_csv('prediction2.csv', index = False)

