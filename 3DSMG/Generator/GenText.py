import torch
from torch import nn, Tensor

class GenText():
    def __init__(self, textures):
        self.textures = textures
    
    def forward(self, texture):
        return (self.textures * texture).sum(-1)
    
    def to(self, device):
        self.textures = self.textures.to(device)
