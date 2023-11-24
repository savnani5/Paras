from torch import nn
import torch.nn.functional as F
import torch
import config


class SobelOperator(nn.Module):
    """Model class implementing Sobel operator."""
    def __init__(self):
        super(SobelOperator, self).__init__()
        Gx = torch.randn(5, 5).unsqueeze(0).unsqueeze(0)
        Gy = torch.randn(5, 5).unsqueeze(0).unsqueeze(0)
        self.weight_gx = nn.Parameter(data=Gx, requires_grad=True)
        self.weight_gy = nn.Parameter(data=Gy, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for Sobel Operator model"""
        x_v = F.conv2d(x, self.weight_gx, padding=2)
        x_h = F.conv2d(x, self.weight_gy, padding=2)
        return torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + config.epsilon)
