# Import necessary libraries
import sqlite3
import pandas as pd

import setup.setup as setup
import EDA.eda_step as EDA
import model_select.model_select as model_select
import model_eval.model_eval as model_eval

(
    db_path,
    target_col,
    redundant_col_list,
    time_col_list,
    standard_col_list,
    encode_col_list,
    model_test_size,
    model_random_state,
    model_num_jobs,
    model_search_method,
    model_cv_num,
    model_param_dict,
) = setup.setup_stg()

# Create connection to SQL database
conn = sqlite3.connect(db_path)

# Get data from 'score' table
score_data_query = "SELECT * FROM score;"
score_data = pd.read_sql_query(score_data_query, conn)


# Using analysis from task_1 EDA, perform data preprocessing and feature scaling
fil_score_data, preprocessor, X_train, X_test, Y_train, Y_test = EDA.ml_eda_step(
    score_data,
    redundant_col_list,
    time_col_list,
    standard_col_list,
    encode_col_list,
    target_col,
    model_test_size,
    model_random_state,
)

# Pre-select a few models and train the models to get the best optimized parameters
best_estimator_dict = model_select.model_selection(
    preprocessor,
    X_train,
    Y_train,
    model_random_state,
    model_search_method,
    model_cv_num,
    model_num_jobs,
    model_param_dict,
)

# Evaluate pre-selected models to get mean-squared error and r^2 values to determine which model is better for current dataset
model_eval.model_evaluation(X_test, Y_test, best_estimator_dict)

# Best model decision -
# - If model variance is priority, look for highest R^2
# - If predictive accuracy is priority, look for lowest Mean Squared Error
