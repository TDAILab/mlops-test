import os
import sys
import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def main():
    print(os.listdir("data/input"))
    df = pd.read_csv(
        "data/input/iris.csv",
        header=None,
        names=["label", "feat1", "feat2", "feat3", "feat4"])

    X = df[["feat1", "feat2", "feat3", "feat4"]].values
    y = df[["label"]].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    clf = KNeighborsClassifier(n_neighbors=5)
    print("fitting...")
    clf.fit(X_train, y_train)

    print("predicting...")
    y_predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_predictions)

    os.makedirs("data/output", exist_ok=True)
    model_path = "data/output/model.pkl"
    pickle.dump(clf, open(model_path, 'wb'))


if __name__ == '__main__':
    main()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
