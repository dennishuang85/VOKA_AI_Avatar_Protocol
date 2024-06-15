from torch import nn, tensor

class SpatialNet(nn.Module):
    def __init__(self):
        super(SpatialNet, self).__init__()
        """self.base = tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]],
                           device = device,
                            requires_grad = False)"""
        self.direct = nn.Linear(3, 12)
        self.dense0 = nn.Linear(3, 512)
        self.dense1 = nn.Linear(512, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 12)
        self.ELU = nn.ELU()
        self.PReLU = nn.PReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x0 = self.ELU(self.dense0(x))
        x1 = self.relu(self.dense1(x0))
        x2 = self.relu(self.dense2(x1))
        x3 = self.relu(self.dense3(x2))
        x4 = self.relu(self.dense4(x3))
        x5 = self.dense5(x4)
        x6 = x5 + self.direct(x)
        ret = x6.view(x6.shape[:-1] + (3, 4))
        return ret
        