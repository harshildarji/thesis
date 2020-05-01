import torch
import torch.nn as nn
import torch.jit as jit
from layer import MaskedRecurrentLayer
from typing import List
from torch import Tensor


class PruneRNN(jit.ScriptModule):
    def __init__(self, input_size, hidden_layers: list, mode='tanh'):
        super(PruneRNN, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.mode = mode

        self.recurrent_layers = nn.ModuleList()
        for l, hidden_size in enumerate(hidden_layers):
            input_size = input_size if l == 0 else hidden_layers[l-1]
            self.recurrent_layers.append(MaskedRecurrentLayer(input_size, hidden_size, mode))

    @jit.script_method
    def forward(self, input):
        # type: (Tensor) -> Tensor
        batch_size = input.size(1)
        inputs = input.unbind(0)

        for l, layer in enumerate(self.recurrent_layers):
            hx = torch.zeros(batch_size, self.hidden_layers[l], dtype=input.dtype, device=input.device)
            outputs = []

            for i in range(len(inputs)):
                hx = layer(inputs[i], hx)
                outputs.append(hx.unsqueeze(0))

            inputs = torch.cat(outputs, dim=0)

        return inputs[-1]
