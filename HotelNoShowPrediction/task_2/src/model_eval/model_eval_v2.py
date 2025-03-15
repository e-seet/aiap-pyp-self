import pandas as pd
import time
import joblib
import os

import setup.duration_cal as duration_cal
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict


# Load and evaluate trained models from .pkl files
def load_and_evaluate_models(model_folder: str, X_test: pd.DataFrame, Y_test: pd.Series):
    eval_result_dict = {}

    # Get all model files in the directory
    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pkl")]

    for model_file in model_files:
        model_start_time = time.time()
        model_path = os.path.join(model_folder, model_file)

        print(f"Loading model: {model_file}")
        
        # Load trained model
        model = joblib.load(model_path)

        print(f"Evaluating {model_file} now...")
        Y_predict = model.predict(X_test)

        # Compute metrics
        confuse_matrix = confusion_matrix(Y_test, Y_predict)
        class_rpt = classification_report(Y_test, Y_predict, output_dict=True)

        # Store results
        eval_result_dict[model_file] = {
            "Confusion Matrix": confuse_matrix,
            "Classification Report": class_rpt,
        }

        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        model_duration, model_tag = duration_cal.duration_cal(model_total_time)

        print(f"{model_file} evaluation completed in {model_duration:.3f} {model_tag}!\n")

    # Convert results to DataFrame
    results_df = pd.DataFrame(eval_result_dict).T
    print(results_df)

    # Save evaluation results to CSV
    results_df.to_csv("model_evaluation_results.csv", index=True)
    print("Evaluation results saved to 'model_evaluation_results.csv'")


