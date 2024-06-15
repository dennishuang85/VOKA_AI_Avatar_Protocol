import torch
from torch import nn, Tensor
from Generator.GenBase import GenBase

class GenMesh(nn.Module):
    def __init__(self, base: GenBase, shapes: GenBase, landmarks: Tensor = None):
        self.base: GenBase = base
        self.shapes: GenBase = shapes
        self.landmarks: Tensor = landmarks
    
    def get_base(self, identity):
        return self.base.get(identity)
    
    def get_shapes(self, identity):
        return self.shapes.get(identity)
    
    def get_landmarks(self, identity, expression = None):
        base = self.base.get(identity, self.landmarks)
        if expression is None:
            return base
        shapes = self.shapes.get(identity, self.landmarks)
        return base + (shapes * expression).sum(-1)
    
    def get(self, identity, expression = None):
        base = self.base.get(identity)
        if expression is None:
            return base
        shapes = self.shapes.get(identity)
        return base + (shapes * expression).sum(-1)
    
    def to(self, device):
        self.base = self.base.to(device)
        self.shapes = self.shapes.to(device)
        self.landmarks = self.landmarks.to(device)
