import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, name, layer_sizes):

        super().__init__()

        self.MLP = nn.Sequential()
        self.name = name

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(f"{name}_fc_L{i}", nn.Linear(in_size, out_size))
            self.MLP.add_module(f"{name}_fc_A{i}", nn.ReLU())
    
    def forward(self, x, p):
        
        out = self.MLP(x.to(torch.float32))
        return out*p

if __name__ == "__main__":
    mlp = MLP('1', [2,128,100])
    print(mlp)