# Student Score Prediction Attempt
---

## Introduction
Personal attempt on the StudentScorePrediction question from AIAP

## Prerequsities
Refer to requirements.txt for required Python libraries <br>

## Overview
### Task 1
Use Jupyter notebook (eda.ipynb) to do Exploratory Data Analysis (EDA) on the provided SQL database (score.db) <br> <br>
Requirements:
- Use SQLite or similar to open and read the SQL database
- Clean up and process data to produce some form of correlation between students' score and the other data sets
- For details on the feature engineering done to the data from the SQL database, please refer to the table in 
- Columns that are not related to the students' score and those that are not correlatable are dropped from the data set
- Using scatterplot and correlation matrix, the relationship between the remaining features and the students' score are shown

#### Summarized Details
1. Use SQLite to connect to SQL database, score.db, and check the structure, column data type and data in 'score' table
2. Get the row_count of 'score' table to assess size of data set
3. Extract 'score' table into a DataFrame using SQLite for further processing
4. Drop irrelevant columns that have no relationship with 'final_test' column
5. Clean up DataFrame of empty cells by dropping the rows
6. Performing labelling for columns that have data with text type (Refer to Feature Engineering Table for mapping)
7. Use sleep_time and wake_time to calculate the number of hours of sleep, then drop sleep_time and wake_time columns
8. Check the summary statistics of the DataFrame to get a sense of the benchmark value of weak students ('final_test' value <  25% metric)
9. Plot scatterplots for each feature data against 'final_test' to check correlation
10. Drop columns that do not any clear correlated data to reduce data set
11. Plot correlation matrix to accurately see the correlation value between the feature data and 'final_test' (Further from 0 = Stronger correlation)


#### Feature Engineering Table
| No. | Column Name | Value | Mapping |
| :-: | :---------: | :---: | :-----: |
|  1  | direct_admission  |         No        | 0 |
|     |                   |         Yes       | 1 |
|  2  |       CCA         |       Sports      | 0 |
|     |                   |        Arts       | 1 | 
|     |                   |        Clubs      | 2 |
|     |                   |        None       | 3 |
|  3  |  learning_style   |       Visual      | 0 |
|     |                   |      Auditory     | 1 |
|  4  |      tuition      |        No/N       | 0 |
|     |                   |       Yes/Y       | 1 |
|  5  | mode_of_transport |        Walk       | 0 |
|     |                   | Public Transport  | 1 |
|     |                   | Private Transport | 2 |
|     |                   |                   |   |

<br>

----

### Task 2
Use Python scripts (*.py) to create an End-to-End Machine Learning pipeline <br> <br>
Requirements:
- Appropriate data processing and feature engineering
- Appropriate use and optimization of at least 3 algorithms/models
- Appropriate explanation for choice of algorithms/models
- Appropriate use of evaluation metrics
- Appropriate explanation of evaluation metrics

User config - Located in cfg/config.yaml

#### Folder Structure
```md
./src
├── cfg
|   └── config.yaml
├── EDA
|   └── eda_step.py
├── model_eval
|   └── model_eval.py
├── model_select
|   └── model_select.py
├── setup
|   └── setup.py
├── main.py
└── requirements.txt
```

#### Pipeline Execution and Parameter Modification Flow
- TODO: Not done

#### Pipeline Flow
- TODO: Not done

#### Model Choice Evaluation
- Linear Regression
    - Parameters: None
    - Evaluation:
        - This model is also a quick and interpretable model that can be used as a benchmark for comparison
- Random Forest Regression
    - Parameters:
        - Number of estimators - Number of decision trees in the random forest
        - Max depth - Number of splits that each decision tree is allowed to make
    - Evaluation:
        - Since the current data set has many features and having insight on the important features, random forest regression is optimal
        - This model has better model accuracy, at the expense of intepretabiity and computational resources available
        - Number of estimators parameter determines the number of decision trees
            - Increasing the parameter generally improves performance up to a certain point before more trees reduce variance by averaging the predictions
            - Ideal method is to start at a value (example: 100) and increase the parameter value until performance stabilizes
        - Max depth parameter is tweaked to limit the depth of each decision tree, which controls the complexity of trees
            - A deeper tree captures more complex patterns but there is a risk of overfitting the training data
            - Ideal method is to start at a value (example: 3) and increase until the performance peaks
- Support Vector Regression (SVR)
    - Parameters:
        - C - Regularization parameter (Regularization strength is inversely proportional to C)
        - Kernel - Kernel type to be used in algorithm
    - Evaluation:
        - As the datasets has a non-linear relationship between the features and 'final_test' and also requires precise predictions, support vector regression is optimal
        - Since the priority in this case is high accuracy but the relationships in the data are not well-understood, this model also stands out as an excellent choice
        - C parameter controls trade-off between achieving low error on training data and minimizing model complexity to improve generalization
            - Higher values allow model to focus more on minimizing training error, while lower values emphasize simpler models that generalize better
            - Ideal method is to start at a value (exmple: 1) and tune across a wide range (example: 0.1 to 200)
        - Kernel parameter defines how the input space is transformed to capture relationships between features and target variable, 'final_test'
            - Different kernels are suited for different types of data and relationships
            - Ideal method is to start with radial basis function (RBF) kernel and switch to other kernels if other relationships are suspected

<br>

## Reference
https://github.com/aisingapore/AIAP-Technical-Assessment-Past-Years-Series/tree/main/StudentScorePrediction