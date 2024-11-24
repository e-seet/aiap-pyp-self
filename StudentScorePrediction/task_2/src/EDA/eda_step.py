import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from typing import List


# score_data DataFrame data preprocessing step
def ml_eda_step(
    score_data: pd.DataFrame,
    redundant_col_list: List,
    time_col_list: List,
    standard_col_list: List,
    encode_col_list: List,
    target_col: str,
    model_test_size: float,
    model_random_state: int,
):
    # Remove duplicate rows based on 'student_id' column data
    score_data = score_data.drop_duplicates(subset="student_id", keep="first")

    ## Drop rows with empty cells & redundant columns
    fil_score_data = score_data.drop(columns=redundant_col_list)

    ## Drop rows if there are any missing values
    fil_score_data = fil_score_data.dropna()

    ## Apply cal_sleep_hours function to score_data
    fil_score_data["sleep_hours"] = fil_score_data.apply(
        lambda row: cal_sleep_hours(row["sleep_time"], row["wake_time"]),
        axis=1,
    )

    ## Drop time_col after calculating sleep_hours
    fil_score_data = fil_score_data.drop(columns=time_col_list)

    ## Standardize data set
    scaler = StandardScaler()
    fil_score_data[standard_col_list] = scaler.fit_transform(
        fil_score_data[standard_col_list]
    )

    ## Perform one-hot encoding on categorical variables
    encoded_score_data = pd.get_dummies(
        fil_score_data, columns=encode_col_list, drop_first=True
    )
    bool_col = encoded_score_data.select_dtypes(include=["bool"]).columns
    encoded_score_data[bool_col] = encoded_score_data[bool_col].astype(int)

    ## Get sorted correlation values of features
    corr_matrix = encoded_score_data.corr()
    final_test_corr = corr_matrix[target_col]
    sorted_corr = final_test_corr.abs().sort_values(ascending=False)

    ## Drop features with correlation value < 0.1
    drop_cols = sorted_corr[sorted_corr < 0.1].index
    final_score_data = encoded_score_data.drop(columns=drop_cols, axis=1)

    ## Prepare model testing and training dataset for both features and target
    ## Prepare numerical data preprocessing
    preprocessor, X_train, X_test, Y_train, Y_test = model_data_prep(
        final_score_data.drop(columns=target_col),
        final_score_data[target_col],
        model_test_size,
        model_random_state,
    )

    return fil_score_data, preprocessor, X_train, X_test, Y_train, Y_test


# Calculate sleep hours based on sleep_time and wake_time
def cal_sleep_hours(sleep_time, wake_time):
    # Convert sleep_time and wake_time into datetime objects
    sleep_time = datetime.strptime(sleep_time, "%H:%M")
    wake_time = datetime.strptime(wake_time, "%H:%M")

    # If wake_time is earlier than sleep_time, assume wake_time is on the next day
    if wake_time < sleep_time:
        wake_time += pd.Timedelta(days=1)

    # Calculate different in hours
    sleep_duration = (
        wake_time - sleep_time
    ).total_seconds() / 3600  # Convert seconds to hours

    return sleep_duration


# Prepare data for model use in later steps
def model_data_prep(
    X_data_df: pd.DataFrame,
    Y_data_df: pd.DataFrame,
    model_test_size: float,
    model_random_state: int,
):
    ## Separate features (X) and target variable (Y)
    X = X_data_df
    Y = Y_data_df

    ## Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=model_test_size, random_state=model_random_state
    )

    ## Preprocessing for numerical data
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), X.columns)]
    )

    return preprocessor, X_train, X_test, Y_train, Y_test
