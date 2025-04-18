import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16, Swin_V2_B_Weights, swin_v2_b, vgg11, VGG11_Weights

class Model1(nn.Module):
    def __init__(self, num_classes=9):
        super(Model1, self).__init__()
        
        weights = ViT_B_16_Weights.DEFAULT
        vitB = vit_b_16(weights=weights)
        
        self.backbone = vitB
        self.backbone.heads = nn.Identity()

        n_features = vitB.hidden_dim

        self.classifier = nn.Linear(n_features, num_classes) 

        self.regressor = nn.Linear(n_features, 4) 

    def forward(self, x):
        out = self.backbone(x)

        class_out = self.classifier(out)
        bbox_out = self.regressor(out) 

        return class_out, bbox_out
    
class Model2(nn.Module):
    def __init__(self, num_classes=9):
        super(Model2, self).__init__()
        
        weights = Swin_V2_B_Weights.DEFAULT
        swin = swin_v2_b(weights=weights)

        n_features = swin.head.in_features

        swin.head = nn.Identity()
        self.backbone = swin

        self.classifier = nn.Linear(n_features, num_classes) 

        self.regressor = nn.Linear(n_features, 4) 

    def forward(self, x):
        out = self.backbone(x)

        class_out = self.classifier(out)
        bbox_out = self.regressor(out) 

        return class_out, bbox_out
    
class Model3(nn.Module):
    def __init__(self, num_classes=9):
        super(Model3, self).__init__()
        
        weights = VGG11_Weights.DEFAULT
        vgg = vgg11(weights=weights)

        self.backbone = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        n_features = 512

        self.classifier = nn.Linear(n_features, num_classes) 

        self.regressor = nn.Linear(n_features, 4) 

    def forward(self, x):
        out = self.backbone(x)
        out = self.pool(out)
        out = torch.flatten(out, 1)

        class_out = self.classifier(out)
        bbox_out = self.regressor(out) 

        return class_out, bbox_out