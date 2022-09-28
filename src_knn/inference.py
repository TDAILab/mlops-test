import pandas as pd


def predict(model, json_input: pd.core.frame.DataFrame[float]) -> float:
    prediction = model.predict(json_input)
    print(f"prediction : {prediction}")
    return prediction
