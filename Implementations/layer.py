import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import init


class MaskedRecurrentLayer(nn.Module):
    """
    Base class for layer initialization.
    (slightly modified version of __init__ of class nn.RNNBase)

    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        mode: Can be either 'RNN_TANH', 'RNN_RELU', 'LSTM' or 'GRU'. Default 'RNN_TANH'
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(self, input_size, hidden_size, mode='RNN_TANH', batch_first=False):
        super(MaskedRecurrentLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode.upper()
        self.batch_first = batch_first

        if self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU':
            gate_size = hidden_size
        elif self.mode == 'GRU':
            gate_size = 3 * hidden_size
        elif self.mode == 'LSTM':
            gate_size = 4 * hidden_size
        else:
            raise ValueError("Unrecognized mode: '{}'".format(mode))

        self.weight_ih = Parameter(torch.randn(gate_size, input_size))
        self.weight_hh = Parameter(torch.randn(gate_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(gate_size))
        self.bias_hh = Parameter(torch.randn(gate_size))
        self.reset_parameters()

        self.register_buffer('mask_i2h', torch.ones((gate_size, self.input_size), dtype=torch.bool))
        self.register_buffer('mask_h2h', torch.ones((gate_size, self.input_size), dtype=torch.bool))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def set_i2h_mask(self, mask):
        self.mask_i2h = Variable(mask)

    def set_h2h_mask(self, mask):
        self.mask_h2h = Variable(mask)

    def forward(self, input, hx):
        if isinstance(hx, tuple):
            hx, cx = hx
        igate = torch.mm(input, (self.weight_ih * self.mask_i2h).t()) + self.bias_ih
        hgate = torch.mm(hx, (self.weight_hh * self.mask_h2h).t()) + self.bias_hh

        if self.mode == 'RNN_TANH':
            return self.__tanh(igate, hgate)
        elif self.mode == 'RNN_RELU':
            return self.__relu(igate, hgate)
        elif self.mode == 'GRU':
            return self.__gru(igate, hgate, hx)
        elif self.mode == 'LSTM':
            return self.__lstm(igate, hgate, hx, cx)

    def __tanh(self, igate, hgate):
        return torch.tanh(igate + hgate)

    def __relu(self, igate, hgate):
        return torch.relu(igate + hgate)

    def __gru(self, igate, hgate, hx):
        i_reset, i_input, i_new = igate.chunk(3, 1)
        h_reset, h_input, h_new = hgate.chunk(3, 1)

        reset_gate = torch.sigmoid(i_reset + h_reset)
        input_gate = torch.sigmoid(i_input + h_input)
        new_gate = torch.tanh(i_new + reset_gate * h_new)

        hx = new_gate + input_gate * (hx - new_gate)
        return hx

    def __lstm(self, igate, hgate, hx, cx):
        gates = igate + hgate

        input_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cx = (forget_gate * cx) + (input_gate * cell_gate)
        hx = out_gate * torch.tanh(cx)
        return hx, cx

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, mode={mode}'

        if self.batch_first:
            s += ', batch_first={batch_first}'

        return s.format(**self.__dict__)
