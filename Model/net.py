import os
import torch
import torch.nn as nn

class Model_FCNet(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super(Model_FCNet, self).__init__()
        self.input_size = input_size
        self._layers = layers
        self._layer_in = nn.Linear(input_size, layers[0])
        self._hidden_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self._output_layer = nn.Linear(layers[-1], output_size)

    def forward(self, inputs):
        output = self._layer_in(inputs)
        for i, h_i in enumerate(self._hidden_layers):
            output = self.activation(h_i(output))
        output = self._output_layer(output)    
        output = self.positive(output)

        return output

    def activation(self, o):

        return torch.tanh(o)

    def positive(self, o):
        if self.input_size == 3:
            output = torch.exp(-o)
        else:
            output = torch.log(1.0 + torch.exp(o))
        return output

# other setting

def Xavier_initi(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

def save_param(net, path):
    torch.save(net.state_dict(), path)


def load_param(net, path):
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    else:
        print("File does not exist.")
