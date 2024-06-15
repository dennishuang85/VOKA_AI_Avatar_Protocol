from torch import nn, tensor, cat

class Deformer(nn.Module):
    def __init__(self, input_size):
        super(Deformer, self).__init__()
        
        self.direct = nn.Linear(input_size, 12)
        self.dense0 = nn.Linear(input_size, 512)
        self.dense1 = nn.Linear(512, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 12)
        self.PReLU = nn.PReLU()
        
    def forward(self, x, identity):
        shape = x.shape
        shape[-1] = -1
        x = cat(x, identity.expan(shape))
        x0 = self.PReLU(self.dense0(x))
        x1 = self.PReLU(self.dense1(x0))
        x2 = self.PReLU(self.dense2(x1))
        x3 = self.PReLU(self.dense3(x2))
        x4 = self.PReLU(self.dense4(x3))
        x5 = self.dense5(x4)
        x6 = x5 + self.direct(x)
        ret = x6.view(x6.shape[:-1] + (3, 4))
        return ret