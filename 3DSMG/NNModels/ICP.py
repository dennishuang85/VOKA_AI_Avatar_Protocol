from torch import nn, tensor
from torch.nn.parameter import Parameter

class ICP(nn.Module):
    def __init__(self, N):
        super(ICP, self).__init__()
        self.trans_mat = Parameter(
            tensor([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]]).repeat(N, 1, 1))
    
    def forward(self, x):
        return self.trans_mat

        