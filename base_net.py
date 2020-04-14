import torch.nn as nn
import torch.nn.functional as F


class MosquitoNet(nn.Module):

    def __init__(self):
        super(MosquitoNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16,kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )


        self.classifier = nn.Sequential(
            nn.Linear(64*15*15,512),
            nn.ReLU(True),
            nn.Dropout2d(p=0.02),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout2d(p=0.02),
            nn.Linear(128,2),

        )


    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        
        
        
        
        return x

