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
    num_map_dict,
    standard_list,
    one_hot_list,
    model_test_size,
    model_random_state,
    model_search_method,
    model_cv_num,
    model_scoring,
    model_num_jobs,
    model_param_dict,
) = setup.setup_stage()

# Create connection to SQL database
conn = sqlite3.connect(db_path)

# Get data from 'noshow' table
noshow_data_query = "SELECT * FROM noshow;"
noshow_data_df = pd.read_sql_query(noshow_data_query, conn)

# Using analysis from task_1 EDA, perform data preprocessing, feature data standardization and one-hot encoding
fil_noshow_data_df, preprocessor, X_train, X_test, Y_train, Y_test = EDA.ml_eda_step(
    noshow_data_df,
    target_col,
    num_map_dict,
    standard_list,
    one_hot_list,
    model_test_size,
    model_random_state,
)

# Pre-select a few models and train models to get best optimized parameters
best_estimator_dict = model_select.model_selection(
    preprocessor,
    X_train,
    Y_train,
    model_random_state,
    model_search_method,
    model_cv_num,
    model_scoring,
    model_num_jobs,
    model_param_dict,
)

# Evaluate pre-selected models to get mean-squared error and r^2 values to determine which model is better for current dataset
model_eval.model_evaluation(X_test, Y_test, best_estimator_dict)

# Best model decision -
## If model variance is priority, look for highest R^2
## If predictive accuracy is priority, look for lowerst MSE (0 == Perfect model)
