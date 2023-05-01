# AIAP Batch 14 Technical Assessment
Name: Ng Kok Woon
Email: kokwoon.ng@gmail.com

# 1. Project Overview
## 1A. Executive Summary
> The objective of the project is to evaluate and identify Machine Learning (ML) models that can help
> fishy & co. to predict if there will be rain the next day. 

Using the data provided by fishy & co, we will first perform a general Exploratory Data Analysis (EDA) 
to gain a better understanding of the weather dataset, before we build a end-to-end Machine Learning Pipeline (MLP)
that will be able to evaluate and recommend the best model based on the user's criterion.

The future iterations of the project would require the application to take in "unseen" data from a specified folder
to compute the predicted labels in an preferred output by the audience for their actual utilization. 

## 1B. Task Summary
- [x] Write the press release
- [ ] 

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

### A. Linux
Clone this project and run the `run.sh` shell bash script in your linux environment with python activated
```
# clone this notebook
git clone https://github.com/ahjimomo/aiap14-ng-kok-woon-685E.git

# run the sh script for mlp pipeline
sh run.sh
```

### B. Windows
Clone this project and run the `run.sh` shell bash script in your Windows environment
```
# clone this notebook
git clone https://github.com/ahjimomo/aiap14-ng-kok-woon-685E.git

# run the sh script
bash run.sh
```
[^1]: I am unable to find the right way to run the `run.sh` executable script, above is based on online research to the best of my knowledge.



