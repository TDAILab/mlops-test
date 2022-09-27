import numpy as np
import pandas as pd
import torch


def predict(model, json_input):
    json_numpy = np.array(json_input).astype(np.float32)
    json_tensor = torch.from_numpy(json_numpy)
    prediction = torch.argmax(model(json_tensor)).numpy()
    print(f"prediction : {prediction}")
    return prediction
