import torch
import torch.nn as nn

class RGBToGrayscale(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False)
        
        # Initialize weights to grayscale conversion coefficients
        # weights = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]])  # shape (1, 3, 1, 1)
        # self.conv.weight.data = weights

        # Freeze if not trainable
        if not trainable:
            self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)  # Output shape: (B, 1, H, W)
        # return x.mean(dim=1, keepdim=True)  # Output shape: (B, 1, H, W)
        # return x[:, 0:1, :, :]  # Output shape: (B, 1, H, W)