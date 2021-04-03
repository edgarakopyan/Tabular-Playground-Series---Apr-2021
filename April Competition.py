import pandas as pd
import os

# Check if directory is correct
os.getcwd()

# Upload our data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Delete names column
