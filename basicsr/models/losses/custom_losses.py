import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

class EdgeLoss(torch.nn.Module):
    '''Sobel-based edge-consistency loss (L1).'''
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                               dtype=torch.float32).view(1,1,3,3)/8
        sobel_y = sobel_x.transpose(2,3)
        self.register_buffer('fx', sobel_x)
        self.register_buffer('fy', sobel_y)

    def forward(self, pred, gt):
        def grad(img, kx, ky):
            gx = F.conv2d(img, kx, padding=1)
            gy = F.conv2d(img, ky, padding=1)
            return (gx.abs() + gy.abs())
        return F.l1_loss(grad(pred,self.fx,self.fy),
                         grad(gt,  self.fx,self.fy))
