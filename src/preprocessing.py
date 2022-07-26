
import argparse
# import boto3
import os
import tarfile
import warnings
import numpy as np
import pandas as pd


def main():
    df = pd.read_csv(
        "data/input/iris.data",
        header=None,
        names=["feat1", "feat2", "feat3", "feat4", "label"])

    X = df[["feat1", "feat2", "feat3", "feat4"]].values
    y = df[["label"]].values.ravel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default='/opt/ml/processing/input')
    parser.add_argument("--output-dir", type=str, default='/opt/ml/processing/output')
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    filename = os.path.join(args.input_dir, 'mnist_png.tgz')

    with tarfile.open(filename, 'r:gz') as t:
        t.extractall(path='mnist')

    prepared_data_path = args.output_dir
    os.makedirs(prepared_data_path, exist_ok=True)

    training_dir = 'mnist/mnist_png/training'
    test_dir = 'mnist/mnist_png/testing'
