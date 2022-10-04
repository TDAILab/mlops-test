import pandas as pd


def predict(model, input: pd.core.frame.DataFrame) -> float:
    prediction = model.predict(input)
    print(f"prediction : {prediction}")
    return prediction
