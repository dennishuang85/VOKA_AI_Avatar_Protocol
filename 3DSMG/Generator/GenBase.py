import torch
from torch import nn, Tensor

class GenBase():
    def __init__(self, base: Tensor, comp: Tensor):
        self.base = base
        self.comp = comp
    
    def get(self, x, s = None):
        if s is None:
            return self.base + (self.comp * x).sum(-1)
        else:
            return self.base[s] + (self.comp[s] * x).sum(-1)
    
    def to(self, device):
        self.base = self.base.to(device)
        self.comp = self.comp.to(device)
