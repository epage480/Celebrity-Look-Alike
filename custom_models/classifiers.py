import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceClassifier(nn.Module):
    def __init__(self, base_model, emb_size, output_classes):
        super().__init__()
        self.base_model = base_model
        self.W = nn.Parameter(torch.Tensor(emb_size, output_classes))
        nn.init.kaiming_uniform_(self.W)
    def forward(self, x):
        x = self.base_model(x)
        # Step 1:
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        # Step 2:
        return x_norm @ W_norm

class SoftMaxClassifier(nn.Module):
    def __init__(self, base_model, emb_size, output_classes):
        super().__init__()
        self.base_model = base_model
        self.fc = nn.Linear(emb_size, output_classes)
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x
