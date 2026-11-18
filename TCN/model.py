import torch
from torch import nn
from torch.nn import functional as F


### Define one convolution layer
## Learn the temporal patterns without looking into future
# incoming = number of input channels (electrodes in first layer, learned features after)
# outgoing = number of leared feature channels to output
# dilation = spacing between three time points used by convolution

class CausalConv(nn.Module):
    def __init__(self, incoming, outgoing, dilation):
        super().__init__()
        self.left = 2 * dilation
        self.conv = nn.Conv1d(incoming, outgoing, 3, dilation=dilation) # kernel size = 3 rn, might need to tune later as a hyperparameter

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0))) # add 0's only to beginning of time dimension


### Stack four of above-define layers
### Convert their feature into predicting movements

class TCN(nn.Module):
    """Input [batch, channels, history]; output [batch, movement dimensions]."""
    def __init__(self, channels, outputs, hidden=64, dropout=0.1):
        super().__init__()
        layers = []
        for incoming, dilation in zip([channels, hidden, hidden, hidden], [1, 2, 4, 8]):
            layers.extend([CausalConv(incoming, hidden, dilation), nn.ReLU(), nn.Dropout(dropout)])
        self.temporal = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        return self.head(self.temporal(x)[:, :, -1])
