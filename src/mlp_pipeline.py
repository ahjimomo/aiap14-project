# ====================================================================================================
# Name          : Ng Kok Woon (Nicky)
# Email         : kokwoon.ng@gmail.com
# File          : mlp.py
# Description   : AIAP Batch 14, Task 2 - MLP Pipeline - main script        
#               
# Content       : 
#                   1. Libraries
#                   2. Importing Dataset
#                   3. Setting Parameters
#                   4. Preprocessing & Feature Engineering
#                   5. Data Preparation
#                   6. Model Fitting & Evaluation
#                   7. Summary


# ====================================================================================================
# 1. Libraries
# ====================================================================================================
# Data Wrangling
import numpy as np 
import pandas as pd
from pandas.api.types import CategoricalDtype
from collections import Counter

# Date/Time
from datetime import datetime
import time

# Statistics/Math
import scipy.stats as ss
import math

# ML Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Databases
import sqlite3

# Custom-defined functions
import preprocessor

# Warnings & Display
import warnings
from pip._internal.operations import freeze
warnings.filterwarnings("ignore")


# ====================================================================================================
# 2. Importing Dataset
# ====================================================================================================
# Print steps
print(f"Hello, welcome to Fishy & Co. Rain Predictor (Beta)!\n"
      f"Please wait while we get the system loaded.\n"
      f"Loading Data...\n")

# Connection to SQLite database
try:
    conn = sqlite3.connect('../data/fishing.db')  
except Exception as err:
    print(f"Connection error:\n{err}")
      
# Extract data as pd.DataFrame
cursor = conn.cursor()
raw_df = pd.read_sql_query('SELECT * FROM fishing', conn)

# Close db connection
conn.close()

# Print before cleaning
print("Data loading completed.\n")
preprocessor.summarize(raw_df, "Raw")


# ====================================================================================================
# 3. Setting Parameters
# ====================================================================================================
random_seed = 42
version = '1.0'


# ====================================================================================================
# 4a. Preprocessing - Removal of duplicated date-location pairs
# ====================================================================================================
# Copy raw_df
cleaned_df = raw_df.copy()

# Remove duplications based on date-locations pair
cleaned_df.drop_duplicates(subset = ['Date', 'Location'], keep = 'first', inplace = True)


# ====================================================================================================
# 4b. Feature Engineering - Verfication and extraction of correctly predicted records
# ====================================================================================================
# Clean values and re-generate values for `RainToday` based on definition (rained if `Rainfall` > 1) 
cleaned_df['RainToday'] = np.where(cleaned_df['Rainfall'] > 1.0, 'Yes', 'No')

# Convert `Date` column to datetime
cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])

# Create column `tmr_rain` to check if there are actual rainfall the next day
cleaned_df['tmr_rain'] = cleaned_df['Date'] + pd.Timedelta(days = 1)

# Generate dataframe based on locations
changi_df = cleaned_df[cleaned_df['Location'] == 'Changi']
woodlands_df = cleaned_df[cleaned_df['Location'] == 'Woodlands']
tuas_df = cleaned_df[cleaned_df['Location'] == 'Tuas']
sentosa_df = cleaned_df[cleaned_df['Location'] == 'Sentosa']

area_lst = [changi_df, woodlands_df, tuas_df, sentosa_df]

# Iterate through each location and check populate `predict_accurate`
for area_df in area_lst:
    # Extract `Date`-`RainToday` as dictionary
    rain_dict = dict(zip(area_df['Date'], area_df['RainToday']))

    # Replace `tmr_rain` column with the value from dictionary, if not found, return NaN
    area_df['tmr_rain'] = area_df['tmr_rain'].map(rain_dict)

    # Create column `predict_accurate` to track records where prediction of `RainTomorrow` had been accurate else inaccurate
    area_df['predict_accurate'] = np.where(area_df['RainTomorrow'] == area_df['tmr_rain'], 'Yes', 'No')

# Join the dataframe back together as filtered_df
cleaned_df = pd.concat([changi_df, woodlands_df, tuas_df, sentosa_df], ignore_index = True)

# Extract records left that are valid that we can use for training
filtered_df = cleaned_df[cleaned_df['predict_accurate'] == 'Yes']


# ====================================================================================================
# 4c. Preprocessing - Standardizing pressure values
# ====================================================================================================
# Standardize pressure values
filtered_df['Pressure3pm'] = filtered_df['Pressure3pm'].str.lower()
filtered_df['Pressure9am'] = filtered_df['Pressure9am'].str.lower()


# ====================================================================================================
# 4d. Preprocessing - Converting Sunshine to positive & drop all duplicates
# ====================================================================================================
# Convert all `Sunshine` values to positive value
filtered_df = preprocessor.convert_pos(filtered_df, 'Sunshine')

# Drop all records with missing values
na_removed_df = filtered_df.dropna()


# ====================================================================================================
# 4e. Feature Engineering - Creating Month Feature
# ====================================================================================================
# Create copy of processed dataframe
prep_df = na_removed_df.copy()

# Extract month from `Date` column
prep_df['month'] = pd.DatetimeIndex(prep_df['Date']).month


# ====================================================================================================
# 5a. Data Preparation - Feature Selection & prepare target labels
# ====================================================================================================
# Extract selected predictors/features & target label from Task 1
processed_df = prep_df[['Sunshine', 'Humidity3pm', 'Cloud3pm', 'Pressure9am', 'Pressure3pm',
                        'WindDir9am', 'WindDir3pm', 'RainTomorrow']]

# Code labels into integer values
#processed_df['RainTomorrow'] = np.where(processed_df['RainTomorrow'] == 'Yes', 1, 0)


# ====================================================================================================
# 5b. Data Preparation - Encoding Categorical-Ordinal Feature (Feature Engineering)
# ====================================================================================================
# Label encode `Pressure9am` and `Pressure3pm`
processed_df['Pressure9am_encode'] = np.where(processed_df['Pressure9am'] == 'low', 0, 
                                              np.where(processed_df['Pressure9am'] == 'med', 1, 2))
processed_df['Pressure3pm_encode'] = np.where(processed_df['Pressure3pm'] == 'low', 0, 
                                              np.where(processed_df['Pressure3pm'] == 'med', 1, 2))

# Dropping original columns from dataset
processed_df.drop(columns = ['Pressure9am', 'Pressure3pm'], inplace = True)


# ====================================================================================================
# 5c. Data Preparation - Downsampling to reduce degree of imbalance
# ====================================================================================================
balanced_df = preprocessor.downsample_df(processed_df, 'RainTomorrow', 'No', 'Yes')

# Review dataset before proceeding
preprocessor.summarize(processed_df, "Processed full data before train-test splitting")
preprocessor.summarize(balanced_df, "Processed class-balanced data before train-test splitting")


# ====================================================================================================
# 5d. Data Preparation - Splitting training and testing set
# ====================================================================================================
# Generate training and testing sets and labels
x_train, y_train, x_test, y_test = preprocessor.prep_train_test(processed_df, 'RainTomorrow', 0.2)
xd_train, yd_train, xd_test, yd_test = preprocessor.prep_train_test(balanced_df, 'RainTomorrow', 0.2)


# ====================================================================================================
# 5e. Data Preparation - Scaling numerical features (Feature Engineering)
# ====================================================================================================
# Define cols with numerical features
num_lst = ['Sunshine', 'Humidity3pm', 'Cloud3pm']

# Scale defined columns & append back to original dataframe: 'robust' applied since we did not remove outliers
x_train, x_test = preprocessor.scaling(x_train, x_test, num_lst, 'robust')
xd_train, xd_test = preprocessor.scaling(xd_train, xd_test, num_lst, 'robust')


# ====================================================================================================
# 5f. Data Preparation - Encoding categorical-nominal features (Feature Engineering)
# ====================================================================================================
# Define categorical features
cat_lst = ['WindDir9am', 'WindDir3pm']

x_train, x_test = preprocessor.b_encode(x_train, x_test, cat_lst)
xd_train, xd_test = preprocessor.b_encode(xd_train, xd_test, cat_lst)
print(f"{'=' * 50}\n"
      f"Thank you for your patience. Fishy & Co. Rain Predictor v{version} is ready!\n")


# ====================================================================================================
# 6a. Building Model - Users' preference and preparation
# ====================================================================================================
# Get user input on cost and haul
criterion, cost, revenue = preprocessor.get_input()
print(f"Thank you, please wait while we evaluate the results.\n"
      f"{'=' * 50}\n")

# Prepare dataframe to store all results
model_names = ['Decision Tree', 'Support Vector Machine', 'Logistic Regression']
full_data_results = pd.DataFrame(index = model_names)
balanced_results = pd.DataFrame(index = model_names)

# Prepare list to store all results
profit_generated, earn_to_potential, model_acc, model_precision, model_recall, model_f1 = [], [], [], [], [], []
performance_lst = [profit_generated, earn_to_potential, model_acc, model_precision, model_recall, model_f1]
bal_profit_generated, bal_earn_to_potential, bal_model_acc, bal_model_precision, bal_model_recall, bal_model_f1 = [], [], [], [], [], []
bal_performance_lst = [bal_profit_generated, bal_earn_to_potential, bal_model_acc, bal_model_precision, bal_model_recall, bal_model_f1]


# ====================================================================================================
# 6b. Building Model - Decision Tree
# ====================================================================================================
# Initialize tree model 
dc_full = DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_depth = 10, random_state = random_seed)
dc_bal = DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_depth = 10, random_state = random_seed)

# Train models
dc_full.fit(x_train, y_train)
dc_bal.fit(xd_train, yd_train)

# Generate predicted labels from models
dc_f_pred = dc_full.predict(x_test)
dc_b_pred = dc_bal.predict(xd_test)

# Generate response and extract results
dc_full_result = preprocessor.print_and_evaluate(y_test, dc_f_pred, 'Decision Tree (full)', cost, revenue)
dc_bal_result = preprocessor.print_and_evaluate(yd_test, dc_b_pred, 'Decision Tree (balanced-class)', cost, revenue)

# Process results into their respective list
preprocessor.parse_results(dc_full_result, performance_lst)
time.sleep(2)
preprocessor.parse_results(dc_bal_result, bal_performance_lst)
time.sleep(2)


# ====================================================================================================
# 6c. Building Model - Support Vector Machine (SVM)
# ====================================================================================================
# Initialize tree model 
svm_full = LinearSVC(penalty = 'l2', max_iter = 200, C = 1, random_state = random_seed)
svm_bal = LinearSVC(penalty = 'l2', max_iter = 200, C = 1, random_state = random_seed)

# Train models
svm_full.fit(x_train, y_train)
svm_bal.fit(xd_train, yd_train)

# Generate predicted labels from models
svm_f_pred = svm_full.predict(x_test)
svm_b_pred = svm_bal.predict(xd_test)

# Generate response and extract results
svm_full_result = preprocessor.print_and_evaluate(y_test, svm_f_pred, 'SVM (full)', cost, revenue)
svm_bal_result = preprocessor.print_and_evaluate(yd_test, svm_b_pred, 'SVM (balanced-class)', cost, revenue)

# Process results into their respective list
preprocessor.parse_results(svm_full_result, performance_lst)
time.sleep(2)
preprocessor.parse_results(svm_bal_result, bal_performance_lst)
time.sleep(2)

# ====================================================================================================
# 6d. Building Model - Logistic Regression
# ====================================================================================================
# Initialize tree model 
lr_full = LogisticRegression(penalty = 'l2', max_iter = 200, C = 1, random_state = random_seed, solver = 'lbfgs')
lr_bal = LogisticRegression(penalty = 'l2', max_iter = 200, C = 1, random_state = random_seed, solver = 'lbfgs')

# Train models
lr_full.fit(x_train, y_train)
lr_bal.fit(xd_train, yd_train)

# Generate predicted labels from models
lr_f_pred = lr_full.predict(x_test)
lr_b_pred = lr_bal.predict(xd_test)

# Generate response and extract results
lr_full_result = preprocessor.print_and_evaluate(y_test, lr_f_pred, 'Logistic Regression (full)', cost, revenue)
lr_bal_result = preprocessor.print_and_evaluate(yd_test, lr_b_pred, 'Logistic Regression (balanced-class)', cost, revenue)

# Process results into their respective list
preprocessor.parse_results(lr_full_result, performance_lst)
time.sleep(2)
preprocessor.parse_results(lr_bal_result, bal_performance_lst)
time.sleep(2)

# ====================================================================================================
# 6e. Building Model - Compilation of results
# ====================================================================================================
# Define column names for each results dataframe
cols = ['profit_generated', 'earning_potential', 'model_acc', 'model_precision', 'model_recall', 'model_f1']
full_data_results = preprocessor.generate_results_df(full_data_results, performance_lst, cols)
balanced_results = preprocessor.generate_results_df(balanced_results, bal_performance_lst, cols)

# Compare results of both table based on average F1-Score
avg_full = full_data_results['model_f1'].mean()
avg_bal = balanced_results['model_f1'].mean()


# ====================================================================================================
# 7. Summary
# ====================================================================================================
print(f"Thank you for your patience, results has been computed.\n\n")

# Compare average F1-score to select table/table to use
if avg_full > avg_bal:
    final_table = full_data_results
else:
    final_table = balanced_results

# Sort table based on select criterion by user
if criterion == 1:
    final_table.sort_values(by = 'earning_potential', ascending = False, inplace = True)
    print(f"Based on your selection to select the model based on (1) Profit generating potential.\n")
elif criterion == 2:
    final_table.sort_values(by = 'model_acc', ascending = False, inplace = True)
    print(f"Based on your selection to select the model based on (2) Model's Accuracy.\n")
else:
    final_table.sort_values(by = 'model_f1', ascending = False, inplace = True)
    print(f"Based on your selection to select the model based on (3) Model's F1-Score.\n")

# Summary
print(f"The Fishy & Co. Rain Predictor (Beta) has selected the following model:\n"
      f"Model: {final_table.index[0]}\n"
      f"Earning with test-set: ${final_table['profit_generated'][0]},000\n"
      f"Profit-to-Potential Ratio: {final_table['earning_potential'][0]}%\n"
      f"Model's accuracy: {final_table['model_acc'][0] * 100}%\n"
      f"Model's precision: {final_table['model_precision'][0] * 100}%\n"
      f"Model's recall: {final_table['model_recall'][0] * 100}%\n"
      f"Model's F1-score: {final_table['model_f1'][0] * 100}%\n")

# End Note
print(f"{'=' * 50}\n"
      f"# END NOTE\n"
      f"{'=' * 50}\n"
      f"Please note that this is a beta-test of the Minimal Viable Product (MVP) for Fishy & Co. rain predictor.\n"
      f"In a fully-developed pipeline with the selection model, the selected model will be pickled for reusability.\n"
      f"An additional pipeline to take the 'unseen' dataset and generate the predicted labels will be built for the working version.\n\n"
      f"Thank you,\n"
      f"Ng Kok Woon (Nicky)\n"
      f"Fishy & Co, AI Engineer")