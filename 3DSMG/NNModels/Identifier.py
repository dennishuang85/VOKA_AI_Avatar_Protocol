from torch import nn, tensor

class Identifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Identifier, self).__init__()
        self.dense0 = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.dense0(x)