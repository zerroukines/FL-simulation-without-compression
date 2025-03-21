import torch
import torch.nn as nn
from torchvision.models import vit_h_14, ViT_H_14_Weights

class Model(nn.Module):
    def __init__(self, learning_rate):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        
        # Load vit_h_14 with pretrained weights (IMAGENET1K_SWAG_LINEAR_V1)
        self.model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        
        # Modify the classification head for CIFAR-10 (10 classes)
        # vit_h_14's head is a Linear layer; we replace it with one for 10 outputs
        num_features = self.model.heads.head.in_features  # 1280 for vit_h_14
        self.model.heads.head = nn.Linear(num_features, 10)
        
        # Loss function (CrossEntropyLoss for classification)
        self.loss_function = nn.CrossEntropyLoss()
        
        # Optimizer (Adam, matching your original MobileNetV2 setup)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        return self.model