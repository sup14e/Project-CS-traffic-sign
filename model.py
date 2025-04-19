import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16, vgg11, VGG11_Weights, resnet50, ResNet50_Weights

class Model1(nn.Module):
    def __init__(self, num_classes=9):
        super(Model1, self).__init__()
        
        weights = ViT_B_16_Weights.DEFAULT
        vit = vit_b_16(weights=weights)
        vit.heads = nn.Identity()  # Remove classification head
        self.backbone = vit
        n_features = vit.hidden_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # Bounding box regressor head
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # Output: [x_min, y_min, x_max, y_max] normalized
            nn.Sigmoid()        # Ensures outputs are in [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        class_out = self.classifier(features)
        bbox_out = self.regressor(features)
        return class_out, bbox_out
    
class Model2(nn.Module):
    def __init__(self, num_classes=9):
        super(Model2, self).__init__()
        
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

class Model3(nn.Module):
    def __init__(self, num_classes=9, dropout_p=0.5):
        super(Model3, self).__init__()

        # Load pretrained VGG11
        weights = VGG11_Weights.DEFAULT
        vgg = vgg11(weights=weights)

        self.backbone = vgg.features  # Convolutional feature extractor

        # Optional: Freeze early layers to retain pretrained features
        # for param in self.backbone[:10].parameters():
        #     param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global avg pool
        n_features = 512

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )

        # Bounding box regressor head
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 4)
        )

        # Apply He initialization to the new layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.backbone(x)
        out = self.pool(out)
        out = torch.flatten(out, 1)

        class_out = self.classifier(out)
        bbox_out = self.regressor(out)

        return class_out, bbox_out

class Model4(nn.Module):
    def __init__(self, num_classes=9):
        super(Model4, self).__init__()
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # till conv5_x
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        n_features = 2048  # output features from resnet50
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Bounding box regressor head
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  # bbox: [x_min, y_min, x_max, y_max]
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        
        class_out = self.classifier(out)
        bbox_out = self.regressor(out)

        return class_out, bbox_out

class Model5(nn.Module):
    def __init__(self, num_classes=9):
        super(Model5, self).__init__()

        # Use ResNet-50 instead of ResNet-18 for better feature extraction
        resnet = resnet50(weights='IMAGENET1K_V1')

        n_features = resnet.fc.in_features
        resnet.fc = nn.Identity()  # Replace the final fully connected layer with Identity

        self.backbone = resnet
        self.classifier = nn.Linear(n_features, num_classes)
        self.regressor = nn.Linear(n_features, 4)

    def forward(self, x):
        out = self.backbone(x)
        class_out = self.classifier(out)
        bbox_out = self.regressor(out)
        return class_out, bbox_out

class Model6(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)

        # Keep layers up to conv5_x block (exclude avgpool and fc)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        n_features = 2048  # Output channels from ResNet50 conv5_x

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Bounding box regressor head
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  # Format: [x_min, y_min, x_max, y_max]
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        class_out = self.classifier(x)
        bbox_out = self.regressor(x)

        return class_out, bbox_out