import numpy as np
import pandas as pd
import torch


def predict(model, input: pd.core.frame.DataFrame[float]) -> np.ndarray[float]:
    input_numpy = np.array(input).astype(np.float32)    # set float type to float32
    input_tensor = torch.from_numpy(input_numpy)
    prediction = np.array([torch.argmax(model(input_tensor))])    # specify output as np.array
    print(f"prediction : {prediction}")
    return prediction
