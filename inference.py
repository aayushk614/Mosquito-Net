import json
import torch
from helper import get_model, get_tensor

model = get_model()
classes=['infected','uninfected']

def get_result(image_bytes):
    tensor = get_tensor(image_bytes)
    tensor = tensor.view(-1, 3, 120, 120)
    outputs = model(tensor)
    predicted = torch.max(outputs, 1)[1]
    result = classes[predicted]
    return predicted, result
