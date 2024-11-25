import pandas as pd
import time

import setup.duration_cal as duration_cal

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from typing import Dict


# Pre-select a few models then execute model training and optimization
# To get the best parameters for model evaluation in the next step
def model_selection(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    model_random_state: int,
    model_search_method: str,
    model_cv_num: int,
    model_scoring: str,
    model_num_jobs: int,
    model_param_dict: Dict,
):
    ## Define models and hyper-parameters
    model_dict = {}

    if "Linear Regression" in model_param_dict:
        model_dict["Linear Regression"] = {
            "model": LinearRegression(),
            "params": model_param_dict["Linear Regression"],
        }
    if "Random Forest" in model_param_dict:
        model_dict["Random Forest"] = {
            "model": RandomForestRegressor(random_state=model_random_state),
            "params": model_param_dict["Random Forest"],
        }
    if "SVR" in model_param_dict:  # Support Vector Regressor
        model_dict["SVR"] = {"model": SVR(), "params": model_param_dict["SVR"]}
    if "MLP Regression" in model_param_dict:
        model_dict["MLP Regression"] = {
            "model": MLPRegressor(random_state=model_random_state),
            "params": model_param_dict["MLP Regression"],
        }
    if "Bayesian Ridge" in model_param_dict:
        model_dict["Bayesian Ridge"] = {
            "model": BayesianRidge(),
            "params": model_param_dict["Bayesian Ridge"],
        }
    if "XG Boost" in model_param_dict:
        model_dict["XG Boost"] = {
            "model": XGBRegressor(
                objective="reg:squarederror", random_state=model_random_state
            ),
            "params": model_param_dict["XG Boost"],
        }

    ## Initialize empty dictionary to store best models
    best_estimators_dict = {}

    ## Loop through each model
    for model_name, mp in model_dict.items():
        model_start_time = time.time()
        print(f"Processing {model_name} now...")
        ### Create pipeline with preprocessing and model
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", mp["model"])]
        )

        if model_search_method == "grid":
            #### Use GridSearchCV for hyper-parameter tuning
            search = GridSearchCV(
                pipeline,
                param_grid=mp["params"],
                cv=model_cv_num,
                scoring=model_scoring,
                n_jobs=model_num_jobs,
            )
        elif model_search_method == "random":
            #### Use RandomizedSearchCV for hyper-parameter tuning
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=mp["params"],
                cv=model_cv_num,
                scoring=model_scoring,
                n_jobs=model_num_jobs,
            )

        search.fit(X_train, Y_train)

        ### Save best model and use parameters for model evaluation
        best_estimators_dict[model_name] = search.best_estimator_
        print(f"Best parameters for {model_name}: {search.best_params_}")

        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        model_duration, model_tag = duration_cal.duration_cal(model_total_time)
        print(f"{model_name} has run tuning for {model_duration:.3f} {model_tag}!")
        print()

    return best_estimators_dict
