import torch
import torch.nn as nn
from torch.nn import _VF
from torch.nn import init
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.nn.parameter import Parameter
 
import math
import numpy as np

def global_prune(model, pruning_perc):
    weights = []
    for param in model.parameters():
        if len(param.data.size()) != 1:
            weights += list(param.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(weights), pruning_perc)

    mask = []
    for param in model.parameters():
        if len(param.data.size()) != 1:
            pruned_inds = param.data.abs() > threshold
            mask.append(pruned_inds.float())

    return mask

def weight_prune(model, pruning_perc, w_type):
    weights = []
    for param, data in model.named_parameters():
        if len(data.size()) != 1 and w_type in param:
            weights += list(data.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(weights), pruning_perc)

    mask = []
    for param, data in model.named_parameters():
        if len(data.size()) != 1 and w_type in param:
            pruned_inds = data.abs() > threshold
            mask.append(pruned_inds.float())

    return mask


class WeightBase(nn.Module):
    """Base class for weight initialization (slightly modified version of __init__ of class nn.RNNBase)"""
    def __init__(self, input_size, hidden_size, batch_first, mode):
        super(WeightBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.mode = mode

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'TANH' or mode == 'RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized mode: '{}'".format(mode))

        self._flat_weights_names = []
        self._all_weights = []

        w_ih = Parameter(torch.Tensor(gate_size, input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        b_hh = Parameter(torch.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)
        param_names = ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']
        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)

        self._flat_weights_names.extend(param_names)
        self._all_weights.append(param_names)
        self._flat_weights = [getattr(self, weight) for weight in self._flat_weights_names]
        self.flatten_parameters()
        self.reset_parameters()

        self.masked = False
        self.i2h_masked = False
        self.h2h_masked = False

    def flatten_parameters(self):
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or torch.backends.cudnn.is_acceptable(any_param):
            return

        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn
            with torch.no_grad():
                torch.__cudnn_rnn_flatten_weight(
                    all_weights, (4), self.input_size, rnn.get_cudnn_mode(self.mode),
                    self.hidden_size, 1, self.batch_first, bool(False))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def set_mask(self, mask):
        self.register_buffer('in_mask', mask[0])
        self.register_buffer('hid_mask', mask[1])
        self.masked = True

    def get_mask(self):
        return Variable(self.in_mask), Variable(self.hid_mask)

    def set_i2h_mask(self, mask):
        self.register_buffer('mask_i2h', mask)
        self.i2h_masked = True

    def get_i2h_mask(self):
        return Variable(self.mask_i2h)

    def set_h2h_mask(self, mask):
        self.register_buffer('mask_h2h', mask)
        self.h2h_masked = True

    def get_h2h_mask(self):
        return Variable(self.mask_h2h)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'

        if self.mode == 'TANH' or self.mode == 'RELU':
            s += ', nonlinearity={mode}'
        elif self.mode == 'LSTM' or self.mode == 'GRU':
            s += ', mode={mode}'

        if self.batch_first:
            s += ', batch_first={batch_first}'

        return s.format(**self.__dict__)

    def forward(self, input, hx):
        if self.masked == True:
            mask_ih, mask_hh = self.get_mask()
            return self.cell(input, self.weight_ih * mask_ih, hx, self.weight_hh * mask_hh)

        if self.i2h_masked == True:
            mask_i2h = self.get_i2h_mask()
            return self.cell(input, self.weight_ih * mask_i2h, hx, self.weight_hh)

        if self.h2h_masked == True:
            mask_h2h = self.get_h2h_mask()
            return self.cell(input, self.weight_ih, hx, self.weight_hh * mask_h2h)

        return self.cell(input, self.weight_ih, hx, self.weight_hh)


class ModuleBase(nn.Module):
    """Base class to forward weights and to set mask"""
    def set_mask(self, pruning_perc):
        mask = global_prune(self, pruning_perc)
        for i, m in enumerate(range(0, len(mask), 2)):
            self.recurrent_layers[i].set_mask((mask[m], mask[m+1]))

    def set_i2h_mask(self, pruning_perc):
        mask = weight_prune(self, pruning_perc, 'ih')
        for i, m in enumerate(mask):
            self.recurrent_layers[i].set_i2h_mask(mask[i])

    def set_h2h_mask(self, pruning_perc):
        mask = weight_prune(self, pruning_perc, 'hh')
        for i, m in enumerate(mask):
            self.recurrent_layers[i].set_h2h_mask(mask[i])

    def forward(self, input):
        batch_size = input.size(0) if self.batch_first else input.size(1)

        for l, hidden_size in enumerate(self.hidden_layers):
            if l == 0: out = input
            hx = torch.zeros(batch_size, hidden_size, dtype=input.dtype, device=input.device)
            out = self.step(self.recurrent_layers[l], out, hx)

        return out[:, -1, :] if self.batch_first else out[-1]


#========== RNN-[TANH, RELU] ==========
class MaskedDeepRNN(ModuleBase):
    """A multi-layer RNN with TANH or RELU non-linearity"""
    def __init__(self, in_features, hidden_layers: list, batch_first=False, nonlinearity='tanh'):
        super(MaskedDeepRNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity

        self.recurrent_layers = nn.ModuleList()
        for l, hidden_size in enumerate(hidden_layers):
            in_size = in_features if l == 0 else hidden_layers[l-1]
            self.recurrent_layers.append(MaskedRNNLayer(in_size, hidden_size, batch_first, nonlinearity))

    def step(self, layer, input, hx):
        in_dim = 1 if self.batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]
            hx = layer(seq, hx)
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)


class MaskedRNNLayer(WeightBase):
    """Individual RNN cell to apply TANH or RELU non-linearity to an input and a hidden state"""
    def __init__(self, input_size, hidden_size, batch_first, nonlinearity='tanh'):
        if nonlinearity == 'tanh':
            self.activation = torch.tanh
        elif nonlinearity == 'relu':
            self.activation = torch.relu
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super(MaskedRNNLayer, self).__init__(input_size, hidden_size, batch_first, mode=nonlinearity.upper())

    def cell(self, input, weight_ih, hx, weight_hh):
        igate = F.linear(input, weight_ih, self.bias_ih)
        hgate = F.linear(hx, weight_hh, self.bias_hh)
        return self.activation(igate + hgate)


#========== GRU ==========
class MaskedDeepGRU(ModuleBase):
    """A multi-layer GRU"""
    def __init__(self, in_features, hidden_layers: list, batch_first=False):
        super(MaskedDeepGRU, self).__init__()
        self.hidden_layers = hidden_layers
        self.batch_first = batch_first

        self.recurrent_layers = nn.ModuleList()
        for l, hidden_size in enumerate(hidden_layers):
            in_size = in_features if l == 0 else hidden_layers[l-1]
            self.recurrent_layers.append(MaskedGRULayer(in_size, hidden_size, batch_first))

    def step(self, layer, input, hx):
        in_dim = 1 if self.batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]
            hx = layer(seq, hx)
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)


class MaskedGRULayer(WeightBase):
    """Individual GRU cell"""
    def __init__(self, input_size, hidden_size, batch_first):
        super(MaskedGRULayer, self).__init__(input_size, hidden_size, batch_first, mode='GRU')

    def cell(self, input, weight_ih, hx, weight_hh):
        igate = F.linear(input, weight_ih, self.bias_ih)
        hgate = F.linear(hx, weight_hh, self.bias_hh)

        i_reset, i_input, i_new = igate.chunk(3, 1)
        h_reset, h_input, h_new = hgate.chunk(3, 1)

        reset_gate = torch.sigmoid(i_reset + h_reset)
        input_gate = torch.sigmoid(i_input + h_input)
        new_gate = torch.tanh(i_new + reset_gate * h_new)

        hx = new_gate + input_gate * (hx - new_gate)
        return hx


#========== LSTM ==========
class MaskedDeepLSTM(ModuleBase):
    """A multi-layer LSTM"""
    def __init__(self, in_features, hidden_layers: list, batch_first=False):
        super(MaskedDeepLSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.batch_first = batch_first

        self.recurrent_layers = nn.ModuleList()
        for l, hidden_size in enumerate(hidden_layers):
            in_size = in_features if l == 0 else hidden_layers[l-1]
            self.recurrent_layers.append(MaskedLSTMLayer(in_size, hidden_size, batch_first))

    def step(self, layer, input, hx):
        in_dim = 1 if self.batch_first else 0
        n_seq = input.size(in_dim)
        cx = hx.clone()
        outputs = []

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]
            hx, cx = layer(seq, (hx, cx))
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)


class MaskedLSTMLayer(WeightBase):
    """Individual LSTM cell"""
    def __init__(self, input_size, hidden_size, batch_first):
        super(MaskedLSTMLayer, self).__init__(input_size, hidden_size, batch_first, mode='LSTM')

    def cell(self, input, weight_ih, hx, weight_hh):
        igate = F.linear(input, weight_ih, self.bias_ih)
        hgate = F.linear(hx[0], weight_hh, self.bias_hh)
        gates = igate + hgate

        input_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cx = (forget_gate * hx[1]) + (input_gate * cell_gate)
        hx = out_gate * torch.tanh(cx)
        return hx, cx