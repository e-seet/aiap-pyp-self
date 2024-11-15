import yaml


def setup_stg():
    # Load configuration file
    with open("cfg/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Accessing configuration settings
    db_path = config["database"]["path"]
    target_col = config["features"]["target"]
    redundant_col_list = config["features"]["redundant_list"]
    non_corr_col_list = config["features"]["non_corr_list"]
    time_col_list = config["features"]["time_list"]
    corr_col_list = config["features"]["corr_list"]
    model_test_size = config["model"]["test_size"]
    model_random_state = config["model"]["random_state"]
    model_name_list = config["model"]["name_list"]
    model_num_jobs = config["model"]["num_jobs"]

    model_param_dict = {}

    if "Linear Regression" in model_name_list:
        model_param_dict["Linear Regression"] = {}

    if "Random Forest" in model_name_list:
        model_param_dict["Random Forest"] = {
            "model__n_estimators": config["rand_forest"]["est_list"],
            "model__max_depth": config["rand_forest"]["depth_list"],
        }

    if "SVR" in model_name_list:
        model_param_dict["SVR"] = {
            "model__C": config["svr"]["c_list"],
            "model__kernel": config["svr"]["kernel_list"],
        }

    if "MLP Regression" in model_name_list:
        model_param_dict["MLP Regression"] = {
            "model__hidden_layer_sizes": config["mlp"]["hidden_layer_size_list"],
            "model__activation": config["mlp"]["activation_list"],
            "model__solver": config["mlp"]["solver_list"],
            "model__learning_rate": config["mlp"]["learning_rate_list"],
            "model_max_iter": config["mlp"]["max_iter_list"],
        }

    if "Bayesian Ridge" in model_name_list:
        model_param_dict["Bayesian Ridge"] = {
            "model__max_iter": config["bayes"]["max_iter_list"],
            "model__alpha_1": config["bayes"]["alpha_1_list"],
            "model__alpha_2": config["bayes"]["alpha_2_list"],
            "model__lambda_1": config["bayes"]["lambda_1_list"],
            "model__lambda_2": config["bayes"]["lambda_2_list"],
        }

    if "XG Boost" in model_name_list:
        model_param_dict["XG Boost"] = {
            "model__n_est": config["xgb"]["n_est_list"],
            "model__learning_rate": config["xgb"]["learning_rate_list"],
            "model__max_depth": config["xgb"]["max_depth_list"],
            "model__subsample": config["xgb"]["subsample_list"],
        }

    return (
        db_path,
        target_col,
        redundant_col_list,
        non_corr_col_list,
        time_col_list,
        corr_col_list,
        model_test_size,
        model_random_state,
        model_num_jobs,
    )
