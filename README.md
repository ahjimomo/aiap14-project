# AIAP Batch 14 Technical Assessment
Name: Ng Kok Woon<br>
Email: kokwoon.ng@gmail.com

# Project Overview
## 1A. Executive Summary
> The objective of the project is to evaluate and identify Machine Learning (ML) models that can help
> fishy & co. to predict if there will be rain the next day. 

Using the data provided by fishy & co, we will first perform a general Exploratory Data Analysis (EDA) 
to gain a better understanding of the weather dataset, before we build a end-to-end Machine Learning Pipeline (MLP)
that will be able to evaluate and recommend the best model based on the user's criterion.

The program provides the flexibility for users to provide the current cost per trip and the gross revenue before cost from each trip,
it also provides options for the users to select the model based on their criterion. For non-technical users, criterion 
such as `profit-earning potential` is provided while for technical users, they can opt for `accuracy` or `F1-score`.

The future iterations of the project would require the application to take in "unseen" data from a specified folder
to compute the predicted labels in an preferred output by the audience for their actual utilization. 

## 1B. Task Summary
- [x] Write the press release
- [ ] 2nd Task

## 1C. File Structure
The file structure of the project is extracted via Windows cmd with `tree /a /f > output.doc`
```
|   .DS_Store
|   .gitignore
|   eda.ipynb
|   output.doc
|   README.md
|   requirements.txt
|   run.sh
|   
+---.github
|   |   .DS_Store
|   |   
|   \---workflows
|           github-actions.yml
|           
+---.ipynb_checkpoints
|       eda-checkpoint.ipynb
|       
+---data
|       fishing.db
|       
|           
\---src
        mlp_pipeline.py
        preprocessor.py
```

## 1D. How to execute
There are 2 recommended ways to run the `mlp_pipeline.py` script.[^1]

### (i) Linux
Clone this project and run the `run.sh` shell bash script in your linux environment with python activated
```
# clone this notebook
git clone https://github.com/ahjimomo/aiap14-ng-kok-woon-685E.git

# run the sh script in Linux Environment
sh run.sh
```

### (ii) Windows
Clone this project and run the `run.sh` shell bash script in your `Windows PowerShell` or `command prompt` terminal.
```
# clone this notebook
git clone https://github.com/ahjimomo/aiap14-ng-kok-woon-685E.git

# run the sh script in terminal
bash run.sh
```
A sample output of the full `mlp_pipeline.py` program run can be found in the [sample_output_log.txt document]("C:\Users\User\Desktop\Nicky\aiap14-ng-kok-woon-685E\sample_output_log.txt").

[^1]: I am unable to find the right way to run the run.sh executable script, above is based on online research to the best of my knowledge.

# Exploratory Data Analysis (EDA)
## 2A. EDA Overview
Before we proceed to use the data, we wanted to gain better insights to the dataset and review any cleaning, preprocessing and/or 
feature-engineering that may be needed.

**We do so by starting out with a set of questions:**
1. What is the size of the dataset we are working with?
2. What is the problem we are dealing with?
3. Does the data meet the 6 dimensions of data quality?
..* Are there any missing values or duplicates? (Completeness & Uniqueness)
..* Is the data correct to the best of our knowledge? (Validity)
..* Is the target label `RainTomorrow` accurate and can we verify it? (Consistency)
..* Are the features correct based on our understanding of what they represent? (Accuracy)
..* Since our data is provided to us, we can ignore the dimension of timeliness 
4. Is there an imbalance between classes that we need to deal with?
5. Features
..* Are the numerical values normally distributed, how do we deal with outliers or missing values?
..* Are the categorical values correct, are there ordinal and nominal data?
..* What are the features that we should use?

## 2B. Findings
Based on our set questions, we performed our EDA in the [JupyterLab Notebook]("C:\Users\User\Desktop\Nicky\aiap14-ng-kok-woon-685E\eda.ipynb"), and
we can present the following findings:

| **Item** | **Description** | **Remark** |
| ---  | :-- | :-- |
| Size of Dataset | 12,997 rows with 21 columns | pre-processed |
| Key identifiers | `Date`-`Location` pair | |
| Duplicates | 1,182 based on our `Date`-`Location` pair | To keep only 1-record each |
| Inconsistency | There are inconsistency for the 3 orders of pressure for the `Pressure9am` and `Pressure3pm` features | `[low, med, high]` |


## 2C. EDA Summary for Task 2
With the findings from the EDA, we can summarize our findings to move forward with the project:

1. **Target Label:**
..* Record's RainTomorrow for Date will be the target label for our project and it should be verified by RainToday label from Date+1 to evaluate the correctness of records
..* There is an uneven distribution skewed towards "No" for `RainTomorrow` and we should consider down-sampling the "No" class to reduce risk of over-fitting

2. **Data Cleaning:**
..* There are Date-Locations pair duplicates that should be removed
..* The `Pressure3pm` and `Pressure9am` categories has to be lowercased to standardise the 3 levels of pressure [low, med, high]
..* There are missing fields in the `RainToday` field and they should be populated based on the `Rainfall` value
..* `Sunshine` feature that represents # hours of bright sunshine contains negative values, assuming that they are incorrectly entered, all values are to be positive
..* After our data cleaning, we should remove records with missing values

3. **Data Pre-Processing/Feature Engineering:**
..* Based on our engineered feature month, we can see that month does have some linear correlation with the target class.
..* From our pie chart visulisation, we can see that our `RainTomorrow` target label is moderately imbalance, and the majority class should be downsampled.

4. **Feature Selection/Encoding:**
..* Applying Pearson's Coefficient for our numerical feature selection, there are no features with strong linear correlation (> 0.5 or < -0.5) with our target label, using a threshold of 0.4, we can select the `Sunshine`, `Humidity3pm`, and `Cloud3pm`
..* Applying Uncertainty Coefficient (Thiel-U) for our categorical feature selection, `Pressure9am` and `Pressure3pm` has strong correlation with our target label, with `WindDir9am` and `WindDir3pm` having moderate strength, we will need to apply encoding to use them for our models.
..* Numerical values except `Rainfall` does not have a large range but mostly are not normally distributed, we can performing scaling for all the numerical features (MLP Pipeline)
..* `Pressure9am` and `Pressure3pm` are ordinal data with the order of low -> high and thus we can perform label-encoding to preserve the ranking
..* The other categorical features for wind direction are nominal data with a larger unique values and thus we should explore MinMax, binary or robust encoding to mitigate the risk for curse of dimensionality

5. **Unrequired Labels:**
..* Based on our project requirement, `ColourOfBoats` should not be a factor to consider for our classification model(s)
..* `Date` and `Location` labels should also not be factors to considered
..* `RainToday` will not be required after it has been used for verification for part (1)

6. **Summary for Machine-Learning Pipeline (MLP):**
..* From the EDA, we can see that the features that we will keep for our project includes `Sunshine`, `Humidity3pm`, `Cloud3pm`, `Pressure9am (Encoding)`, `Pressure3pm (Encoding)`, `WindDir9am (Encoding)`, `WindDir3pm (Encoding)` and `RainTomorrow (Target Class)` to tackle the challenge as a *binary classification problem*


