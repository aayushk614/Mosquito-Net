import io
import torch
from model_st import MosquitoNet
from PIL import Image
import torchvision.transforms as transforms

def get_model():
    checkpoint_path = 'model_param/model_val.pt'
    model = MosquitoNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def get_tensor(image_bytes):
    t_trans = transforms.Compose([
        transforms.Resize((120,120)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open(io.BytesIO(image_bytes))
    return t_trans(img)