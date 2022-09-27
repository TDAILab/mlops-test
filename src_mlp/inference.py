import numpy as np
import pandas as pd
import torch


def predict(model, json_input):
    json_numpy = np.array(json_input).astype(np.float32)    # set float type to float32
    json_tensor = torch.from_numpy(json_numpy)
    prediction = np.array([torch.argmax(model(json_tensor))])    # specify output as np.array
    print(f"prediction : {prediction}")
    return prediction
