import pandas as pd


def predict(model, json_input):
    prediction = model.predict(json_input.astype(float))
    print(f"prediction : {prediction}")
    return prediction
