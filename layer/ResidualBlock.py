import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        
        super(ResidualBlock, self).__init__()
        
        self.ResidualBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),            
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        )
        
    def forward(self, x):
        
        return x + self.ResidualBlock(x)
        