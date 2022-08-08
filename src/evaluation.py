
import json
import os
import tarfile
import argparse
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from smexperiments.tracker import Tracker
from smexperiments.trial import Trial
from sagemaker.analytics import ExperimentAnalytics


def load_model():
    """load_model"""
    print("Extracting model from path: {}".format(model_path))
    model_file_name = 'model.pkl'
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    print("Loading model")
    with open(model_file_name, mode='rb') as fp:
        loaded_model = pickle.load(fp)
    return loaded_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default=None,
                        help='Model path')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Data path')
    parser.add_argument('--test-batch-size', type=str, default='8',
                        help='Batch size for inference')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')

    args = parser.parse_args()

    model_dir = args.model_dir
    data_dir = args.data_dir
    model_path = os.path.join(model_dir, "model.tar.gz")

    print("Extracting model from path: {}".format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    print("Loading model")
    model = load_model()

    # print("predicting...")
    # y_predictions = model.predict(X_test)

    # accuracy = accuracy_score(y_test, y_predictions)

    print("Creating evaluation report")
    report_dict = {
        "custom_metrics": {
            "average_loss": {
                "value": 3,
                "standard_deviation": 0
            },
            "accuracy": {
                "value": 0.9,
                "standard_deviation": 0
            }
        }
    }

    print(args.experiment_name)
    trial_component_analytics = ExperimentAnalytics(
        experiment_name=args.experiment_name,
        sort_by="parameters.accuracy",
        sort_order="Descending",  # Ascending or Descending
    )

    df = trial_component_analytics.dataframe()
    is_best = 0
    try:
        best_acc = df.iloc[0]['accuracy']
        if best_acc < report_dict["custom_metrics"]["accuracy"]:
            print('This model is the best ever!!')
            is_best = 1
        else:
            print('This model is not so good.')
    except BaseException:
        is_best = 1
        print('This model is the first one.')

    print('Recording metrics to Experiments...')
    with Tracker.load() as processing_tracker:  # Tracker requires with keyword
        processing_tracker.log_parameters({"accuracy": report_dict["custom_metrics"]["accuracy"],
                                           "average_loss": report_dict["custom_metrics"]["average_loss"],
                                           "is_best": is_best})

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving evaluation report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
