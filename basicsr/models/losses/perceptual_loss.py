# basicsr/models/losses/perceptual_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.transforms import functional as TF

class PerceptualLoss(nn.Module):
    """VGG19 relu3_3 feature L1 loss."""
    def __init__(self):
        super().__init__()
        # load pretrained VGG19 feature extractor
        vgg_features = vgg19(weights='IMAGENET1K_V1').features.eval()
        # take layers up to and including relu3_3 (which is index 15)
        self.slice = nn.Sequential(*[vgg_features[i] for i in range(16)])
        for p in self.slice.parameters():
            p.requires_grad_(False)
        # imagenet mean/std
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def preprocess(self, x):
        # if single-channel, replicate to 3
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        return (x - self.mean) / self.std

    def forward(self, pred, ref):
        p = self.preprocess(pred)
        r = self.preprocess(ref)
        fp = self.slice(p)
        fr = self.slice(r)
        return F.l1_loss(fp, fr)
