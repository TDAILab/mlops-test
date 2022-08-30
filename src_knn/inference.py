import pandas as pd


def predict(model, json_input):
    prediction = model.predict(json_input)
    print(f"prediction : {prediction}")
    return prediction
