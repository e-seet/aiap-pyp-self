import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from typing import Dict


# Pre-select a few models then execute model training and optimization
# To get the best parameters for model evaluation in the next step
def model_selection(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    model_random_state: int,
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
        # Create pipline with preprocessing and model
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", mp["model"])]
        )

        # Use GridSearchCV for hyper-parameter tuning
        grid = GridSearchCV(
            pipeline,
            param_grid=mp["params"],
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=model_num_jobs,
        )
        grid.fit(X_train, Y_train)

        # Save best model and use the parameters for model evaluation
        best_estimators_dict[model_name] = grid.best_estimator_
        print(f"Best parameters for {model_name}: {grid.best_params_}")

    return best_estimators_dict
