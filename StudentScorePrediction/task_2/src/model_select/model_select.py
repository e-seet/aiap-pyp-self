import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from typing import List


# Pre-select a few models then execute model training and optimization
# To get the best parameters for model evaluation in the next step
def model_selection(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    model_random_state: int,
    rand_forest_est_list: List,
    rand_forest_depth_list: List,
    svr_c_list: List,
    svr_kernel_list: List,
):
    ## Define models and hyper-parameters
    models = {
        "Linear Regression": {"model": LinearRegression(), "params": {}},
        "Random Forest": {
            "model": RandomForestRegressor(random_state=model_random_state),
            "params": {
                "model__n_estimators": rand_forest_est_list,
                "model__max_depth": rand_forest_depth_list,
            },
        },
        "SVR": {  # Support Vector Regressor
            "model": SVR(),
            "params": {
                "model__C": svr_c_list,
                "model__kernel": svr_kernel_list,
            },
        },
    }

    ## Initialize empty dictionary to store best models
    best_estimators_dict = {}

    ## Loop through each model
    for model_name, mp in models.items():
        # Create pipline with preprocessing and model
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", mp["model"])]
        )

        # Use GridSearchCV for hyper-parameter tuning
        grid = GridSearchCV(
            pipeline, param_grid=mp["params"], cv=5, scoring="neg_mean_squared_error"
        )
        grid.fit(X_train, Y_train)

        # Save best model and use the parameters for model evaluation
        best_estimators_dict[model_name] = grid.best_estimator_
        print(f"Best parameters for {model_name}: {grid.best_params_}")

    return best_estimators_dict
