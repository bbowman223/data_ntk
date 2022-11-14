import torch.nn as nn


class FeedforwardLayer(nn.Module):
    def __init__(self, input_d, output_d, act, bias = True):
        super(FeedforwardLayer, self).__init__()
        self.act = act
        self.input_d = input_d
        self.linear = nn.Linear(input_d, output_d, bias=bias)
#         self.bias = torch.tensor(np.random.randn(output_d), requires_grad=True)
        self.linear.weight.data.normal_(mean=0, std=1)
        if bias:
            self.linear.bias.data.normal_(mean=0, std=1)

        
    def forward(self, x):
        x = self.linear(x)
        x = x/self.input_d**0.5
#         x = x/self.input_d**0.5+self.bias/self.input_d**0.5
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

