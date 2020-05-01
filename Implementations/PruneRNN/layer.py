import torch
import torch.jit as jit
from torch.nn import Parameter
from torch import Tensor


class MaskedRecurrentLayer(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, mode='TANH'):
        super(MaskedRecurrentLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode.lower()

        if self.mode == 'tanh':
            gate_size = hidden_size
            self.activation = torch.tanh
        elif self.mode == 'relu':
            gate_size = hidden_size
            self.activation = torch.relu
        elif self.mode == 'gru':
            gate_size = 3 * hidden_size
        elif self.mode == 'lstm':
            gate_size = 4 * hidden_size
        else:
            raise ValueError("Unrecognized mode: '{}'".format(mode))

        self.weight_ih = Parameter(torch.randn(gate_size, input_size))
        self.weight_hh = Parameter(torch.randn(gate_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(gate_size))
        self.bias_hh = Parameter(torch.randn(gate_size))

    @jit.script_method
    def forward(self, input, hx):
        # type: (Tensor, Tensor) -> Tensor
        igate = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        hgate = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        return self.activation(igate + hgate)
