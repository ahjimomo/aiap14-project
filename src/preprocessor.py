# ====================================================================================================
# Name          : Ng Kok Woon (Nicky)
# Email         : kokwoon.ng@gmail.com
# File          : processor.py
# Description   : AIAP Batch 14, Task 2 - MLP Pipeline - Supporting Functions          
#               
# Content       : 
#                   1. Libraries
#                   2. Parameters and Settings
#                   3. Display and Data Selection
#                   4. Preprocessing
#                   5. Scaling
#                   6. Categorical Encoding
#                   7. Post-processing


# ====================================================================================================
# 1. Libraries
# ====================================================================================================
# Data Wrangling
import numpy as np 
import pandas as pd
from collections import Counter

# Statistics/Math
from scipy.stats import kstest

# Encoders/Preprocessing
import category_encoders as ce
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Performance evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics

# Warnings & Display
import warnings
warnings.filterwarnings("ignore")


# ====================================================================================================
# 2. Parameters and Settings
# ====================================================================================================
# Setting random seed for project reproducibility
np.random.seed(42)

# Other potential random_seed required functions
random_seed = 42


# ====================================================================================================
# 3. Display and Data Selection
# ====================================================================================================
def summarize(df, title):
    """
    Support function to provide generic summarized information about dataframe

    :returns: None
    """
    print(f"{title} Dataset:\n"
          f"The dataframe has a total of {df.shape[0]} rows and {df.shape[1]} columns.\n"
          f"Count of duplicated rows: {df.duplicated().sum()}\n"
          f"Presence of rows with missing values: {df.isna().sum().any()}\n"
          f"{'=' * 50}\n")

    return


def print_and_evaluate(y_test, y_pred, title, cost = 1, haul = 2):
    """


    """
    # General information
    print(f"<{'-'*5} Summary for model {title} {'-'*5}>\n")
    print(f"Confusion Report:\n{classification_report(y_test, y_pred)}\n")
    

    # Extract number of values
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    accurate_no_rain = cm[0][0] # Goes haul and didn't rain (+$ | +$)
    false_rain = cm[0][1]       # Could go haul but didn't (- | +$)
    false_no_rain = cm[1][0]    # Went haul but it rained (-$ | -)
    accurate_rain = cm[1][1]    # Didn't go haul and rained (- | -)

    # Calculate assumed profit/loss
    total_potential = (accurate_no_rain + false_rain) * (haul - cost)
    wasted = false_no_rain * cost
    total_earned = (accurate_no_rain * (haul - cost)) - wasted
    accurate_save = accurate_rain * cost
    earn_to_potential = np.round(total_earned/total_potential * 100, 2)

    # Summary in business terms
    print(f"\nBusiness Summary:\n\n"
          f"Based on the assumption that each fishing trip costs ${cost},000 and each successful haul generates ${haul},000 for fishy & co.\n"
          f"We developed {title} model for our rain prediction system:\n"
          f"Accurate Predictions where there are rain: {(accurate_rain/len(y_test) * 100):.2f}%, enabling us to avoid making a loss of ${accurate_save},000\n"
          f"Inaccurately predicted that there won't be rain: {(false_no_rain/len(y_test) * 100):.2f}%, causing us to lose ${wasted},000 \n"
          f"Profit if model is 100% accurate: ${total_potential},000 \n"
          f"Actual profit made (after loss): ${total_earned},000 or {(total_earned/total_potential * 100):.2f}% of total potential earnings.\n"
          f"{'=' * 50}\n\n")

    # Extract model performances
    model_accuracy = np.round(metrics.accuracy_score(y_test, y_pred), decimals = 2)
    model_precision = np.round(metrics.precision_score(y_test, y_pred, average = 'weighted'), decimals = 2)
    model_recall = np.round(metrics.recall_score(y_test, y_pred, average = 'weighted'), decimals = 2)
    model_f1 = np.round(metrics.f1_score(y_test, y_pred, average = 'weighted'), decimals = 2)

    return [total_earned, earn_to_potential, model_accuracy, model_precision, model_recall, model_f1]


def get_input():
    # Get user input on cost and haul
    try:
        cost_trip = abs(int(input("What is the cost per fishing trip? ($'000):\n")))
    except ValueError as v_err:
        cost_trip = 1
        print(f"Your input is not a integer value. We will proceed with $1,000 cost per trip\n")

    try:
        earn_trip = abs(int(input("What is the revenue (gross) from the haul per fishing trip? ($'000):\n")))
    except ValueError as v_err:
        earn_trip = 2
        print(f"Your input is not an integer value. We will proceed with $2,000 gross generated per trip\n")

    if cost_trip > earn_trip:
        print(f"Based on your input, you will be making a loss of ${earn_trip - cost_trip},000 per trip.\n"
              f"Defaulting to $1,000 for cost and $2,000 for revenue per trip.\n")
        cost_trip = 1
        earn_trip = 2

    # Informing user of choices
    print(f"Based on your input that each trip cost ${cost_trip},000 and will make us ${earn_trip},000,\n"
          f"each successful fishing trip will make us a profit of ${earn_trip - cost_trip},000.\n")

    # Get preference for preferred evaluation criterion
    try:
        criterion = int(input(f"What do you wish to pick the model based on? (Select a nuumber)\n"
                              f"\t1. Profit-generating potential\n"
                              f"\t2. Model's Accuracy\n"
                              f"\t3. Model's F1-Score\n"))
    except ValueError as v_err:
        criterion = 1 
        print(f"Your input is not a valid integer option. We will proceed with 1. Profit earned\n")

    return criterion, cost_trip, earn_trip


# ====================================================================================================
# 4. Preprocessing
# ====================================================================================================
def convert_pos(df, col):
    """
    Convert all values to positive values

    :params:
        :df - pd.DataFrame
        :col - String name of column in pd.DataFrame

    :returns:
        :result_df - pd.DataFrame with processed column
    """
    df[col] = df[col].apply(abs)
    return df


def downsample_df(df, col, major, minor):
    """
    Performs downsampling of major class to reduce degree of imbalance between 2 target classes, function does not change original dataframe

    :params:
        :df - Original dataframe with imbalance class population
        :col - String column name of target class
        :major - variable name for major class
        :minor - variable name for minor class

    :returns:
        :balance_df - Dataframe with balanced data of both target classes
    """
    # Prepare a set of downsampled data for comparison
    min_class = df[df[col] == minor]      # minor class (yes rain)
    maj_class = df[df[col] == major]      # majority class (no rain)

    # Resample the major class
    maj_down = resample(maj_class, replace = True,
                        n_samples = len(min_class),
                        random_state = random_seed)

    # Generate new_df with new size
    balance_df = pd.concat([maj_down, min_class])
    return balance_df


def prep_train_test(df, target_col, tsize):
    """
    Function to extract the training and testing set from original dataframe, stratify split with shuffling will be applied.

    :params:
        :df - original pd.DataFrame
        :target_col - String of target label column name
        :tsize - testing size
    
    :returns:
        :x_train - training features
        :x_test - testing features
        :y_train - training true labels
        :y_test - testing true labels
    """
    # Extract features and target label as separate dataframe
    y = df[target_col]
    x = df.drop(columns = [target_col], errors = 'ignore', axis = 1)

    # Split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = tsize,
                                                        random_state = random_seed,
                                                        shuffle = True,
                                                        stratify = y)

    print(f"Splitting Test and training data...\n"
          f"Number of data in training set: {len(x_train)}\n"
          f"Number of data in testing set: {len(x_test)}\n")

    return x_train, y_train, x_test, y_test


# ====================================================================================================
# 5. Scaling
# ====================================================================================================
def check_distribution(arr):
    """
    Perform goodness-if-fit test on numerical array follows the Gaussian (normal distribution) 
    based on Kolmogorov-Smirnov test, usually used for larger sample size (n >= 50),
    with a confidence interval of 95%

    :params:
        :arr - Array of numerical values to test for normal distribution

    :returns:
        :normal - Bool of True if array is assumed to be normally distributed else False
    """
    try: 
        result = kstest(arr, 'norm')

        # If p-value < 0.05, we can reject the null hypothesis of the test and accept the alternative that the data does not follow a normal distribution
        if result[1] < 0.05:
            return False
        else:
            return True

    except Exception as err:
        print(f"Error with KS Test:\n{err}")

    return


def scaling(train_df, test_df, cols, option = 'robust'):
    """
    Function to perform scaling across all specified columns, by default tests if array is normally distributed before selecting type of scaler.

    :params:
        :df - pd.DataFrame containing all data to be scaled
        :cols - List of columns that will be iterated for scaling
        :option - 'robust' for robust scaler, and 'minmax' for MinMax scaler

    :returns:
        :scaled_train - Training dataset with specified columns scaled
        :scaled_test - Testing dataset with specified columns scaled
    """
    try:
        if option == 'robust':
            scaler = RobustScaler()
        elif option == 'minmax':
            scaler = MinMaxScaler()

        # Apply scaling on training & testing dataset
        train_df[cols] = scaler.fit_transform(train_df[cols])
        test_df[cols] = scaler.transform(test_df[cols])

    except Exception as err:
        print(f"Error in scaling:\n{err}")

    return train_df, test_df


# ====================================================================================================
# 6. Categorical Encoding
# ====================================================================================================
def b_encode(train_df, test_df, cols):
    """
    Perform binary encoding for categorical feature and append back to dataframe, to reduce the dimensionality, before dropping off original categorical dataframe

    :params:
        :df - pd.DataFrame
        :cols - List containing specified columns' name
    """
    try:
        # Initialize encoder
        encoder = ce.BinaryEncoder(cols = cols)

        # Fit & Transform for training set for encoded dataframe, append to original dataframe & drop column
        train_df = encoder.fit_transform(train_df)
        test_df = encoder.transform(test_df)

        # Same process with testing set
        #test_encoded = encoder.transform(test_df[col]) 
        #pd.concat([test_df, test_encoded], axis = 1)
        #test_df.drop(columns = [col], inplace = True)

    except Exception as err:
        print(f"Error in categorical encoding:\n{err}")

    return train_df, test_df


# ====================================================================================================
# 7. Postprocessing
# ====================================================================================================
def parse_results(results, storage):
    """
    Process and store results into respective list after predicting

    :params:
        :results - extracted list storing model's performance
        :storage - list of lists used to store model's performance

    :returns: None
    """
    for i in range(len(results)):
        storage[i].append(results[i])

    return


def generate_results_df(df, storage, cols):
    """
    Process and extract results to parse into result pd.DataFrame

    :params:
        :df - pd.Dataframe to store all output
        :storage - list of lists used to store model's performance
        :cols - columns to store each result

    :returns:
        :df - pd.DataFrame with updated result values
    """
    for i in range(len(cols)):
        df[cols[i]] = storage[i]

    return df


