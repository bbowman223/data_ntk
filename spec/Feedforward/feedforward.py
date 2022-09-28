import torch.nn as nn


class FeedforwardLayer(nn.Module):
    def __init__(self, input_d, output_d, act, bias = True):
        super(FeedforwardLayer, self).__init__()
        self.act = act
        self.linear = nn.Linear(input_d, output_d, bias=bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x


class Feedforward(nn.Module):
    def __init__(self, width, input_d, output_d, act, num_layers, bias = True):
        super(Feedforward, self).__init__()
        self.first_layer = FeedforwardLayer(input_d, width, act, bias=bias)
        self.internal_layers = nn.ModuleList([FeedforwardLayer(width, width, act, bias=bias) for _ in range(num_layers - 2)])
        self.output_layer = FeedforwardLayer(width, output_d, act=nn.Identity(), bias=True)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.internal_layers:
            x = layer(x)         
        x = self.output_layer(x)
        return x

