import pandas as pd
import os
import numpy as np
import sklearn.preprocessing as skp
import sklearn.linear_model
import sklearn.ensemble
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Add categorical variables

for i in ['Embarked', 'Cabin_Letter', 'Pclass']:
        train_df= pd.concat([train_df,pd.get_dummies(train_df[i], prefix = "A")], axis = 1)
        test_df = pd.concat([test_df, pd.get_dummies(test_df[i], prefix = "A")], axis = 1)


# Try Logistic and Random Forest again
# Logistic model
model2 = sklearn.linear_model.LogisticRegression().fit(train_df[['Pclass', 'SibSp', 'Parch', 'female', 'C', 'Q', 'S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']], train_df.Survived) # 'Pclass',
prediction3_df = pd.Series(model2.predict(test_df[['Pclass', 'SibSp', 'Parch',  'female', 'C', 'Q', 'S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']]))
prediction3_df = pd.concat([Test_IDs, prediction3_df], axis = 1)
prediction3_df = pd.DataFrame(prediction3_df)
prediction3_df.columns = ['PassengerId', 'Survived']
prediction3_df.to_csv('prediction3.csv', index = False)


# Random Forest
forest2 = sklearn.ensemble.RandomForestClassifier(n_estimators = 125).fit(train_df[['Pclass', 'SibSp', 'Parch', 'female', 'C', 'Q', 'S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']],
                                                                          train_df.Survived)
prediction4_df = forest2.predict(test_df[['Pclass', 'SibSp', 'Parch', 'female', 'C', 'Q', 'S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']])
prediction4_df = pd.DataFrame(prediction4_df)
prediction4_df = pd.concat([Test_IDs, prediction4_df], axis = 1)
prediction4_df.columns = ['PassengerId', 'Survived']
prediction4_df.to_csv('prediction4.csv', index = False)

# Now we separate datas with and without Age and use Age as well
train_df_age = train_df[train_df.Age.notna()]
test_df_age = test_df[test_df.Age.notna()]
test_id_age = Test_IDs[test_df[test_df.Age.notna()].index]
# Run Random Forest

forest3 = sklearn.ensemble.RandomForestClassifier(n_estimators = 500).fit(train_df_age[['A_1', 'A_2', 'A_3', 'Age', 'SibSp', 'Parch', 'female', 'A_C', 'A_Q', 'A_S', 'A_A', 'A_B', 'A_C', 'A_D', 'A_E', 'A_F', 'A_G','A_T']],
                                                                          train_df_age.Survived)
prediction5_df = forest3.predict(test_df_age[['A_1', 'A_2', 'A_3', 'Age', 'SibSp', 'Parch', 'female', 'A_C', 'A_Q', 'A_S', 'A_A', 'A_B', 'A_C', 'A_D', 'A_E', 'A_F', 'A_G','A_T']])
prediction5_df = pd.DataFrame(prediction5_df)
test_id_age = test_id_age.reset_index()
test_id_age = test_id_age.drop(['index'], axis = 1)
prediction5_df = pd.concat([test_id_age, prediction5_df], axis = 1)
prediction5_df.columns = ['PassengerId', 'Survived']
prediction5_df = pd.concat([prediction4_df[prediction4_df.index.isin(test_df[test_df.Age.isna()].index)], prediction5_df], axis = 0)
prediction5_df.to_csv('prediction5.csv', index = False)

# Look at the importance of each feature
for feature, i in zip(['A_1', 'A_2', 'A_3', 'Age', 'SibSp', 'Parch', 'female', 'A_C', 'A_Q', 'A_S', 'A_A', 'A_B', 'A_C', 'A_D', 'A_E', 'A_F', 'A_G','A_T'], forest3.feature_importances_):
    print(feature, i)
# Most important are Age and whether the person is fe(male).
# We can also see, that addition of Age did not significantly improve the prediction performance.

# We now try PCA dimensionality reduction on our data. For this, we first combine the test and train data
# to make sure that columns we get in the end are the same in both datasets

train_df = train_df.reset_index(drop = True)
test_df = test_df.reset_index(drop = True)
test_df.insert(loc = 0, column = 'Survived', value= np.nan)
total_df = pd.concat([train_df.drop(['PassengerId'], axis = 1), test_df]) # , axis = 0, ignore_index=True
total_df = total_df.reset_index(drop=True)
# Now remove the survived column and remove non-float columns
Survived_PCA = total_df.Survived
total_df = total_df.drop(['Survived', 'Cabin', 'Embarked', 'Ticket', 'Pclass', 'Cabin_Letter', 'Cabin_Number'], axis= 1)
# Before we proceed we impute the Age data
total_df_age = total_df[~total_df.Age.isna()]
total_df_age = total_df_age.reset_index(drop=True)
Agemodel = sklearn.linear_model.LinearRegression().fit(total_df_age[['SibSp', 'Parch', 'female']], total_df_age.Age)
Agemodel.score(total_df_age[['SibSp', 'Parch', 'female']], total_df_age.Age)
# This only predicts 2% of the age
# Another option would be to just drop the column
total_df = total_df.drop(['Age', 'Fare'], axis = 1)

# And now we do PCA
pca = PCA()
pca.fit(total_df)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.scatter( range(0, len(pca.explained_variance_ratio_)), cumsum)
plt.show()

# As we see, with 7 features we can explain more than 95% of the variance

pca = PCA(n_components= 7)
total_df_new = pca.fit_transform(total_df)
total_df_new = pd.DataFrame(total_df_new, columns= ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'])

# And we add back in the survived column

total_df_new = pd.concat([total_df_new, Survived_PCA], axis = 1)

# And we separate again into train and test data

train_df2 = total_df_new.iloc[:100000, :]
test_df2 = total_df_new.iloc[100000:, :]
test_df2 = test_df2.drop(['Survived'], axis = 1)

# Now let's retry the Logistic model
X_train, X_test, y_train, y_test = train_test_split(train_df2[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']], train_df2.Survived, test_size=0.33, random_state=42)
forest3 = sklearn.ensemble.RandomForestClassifier(n_estimators = 500, max_depth = 10).fit(X_train, y_train)
prediction7_df = forest3.predict(X_test)
score = forest3.score(X_test, y_test)
score
forest3.feature_importances_

