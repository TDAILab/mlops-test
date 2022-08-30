import numpy as np
import pandas as pd
import torch


def predict(model, json_input):
    json_numpy = np.array(json_input)
    json_tensor = torch.from_numpy(json_numpy).float()
    prediction = torch.argmax(model(json_tensor))  # model:ModelService
    print(f"prediction : {prediction}")
    return prediction
