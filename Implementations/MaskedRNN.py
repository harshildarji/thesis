import torch
import torch.nn as nn
from torch.nn import _VF
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy

def weight_prune(model, pruning_perc):
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


class MaskedDeepRNN(nn.Module):
    def __init__(self, in_features, hidden_layers: list, batch_first=False, mode='tanh'):
        super(MaskedDeepRNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.batch_first = batch_first
        self.mode = mode.lower()

        self.recurrent_layers = nn.ModuleList()
        for l, hidden_size in enumerate(hidden_layers):
            in_size = in_features if l == 0 else hidden_layers[l-1]
            self.recurrent_layers.append(MaskedRecurrentLayer(input_size=in_size, hidden_size=hidden_size, mode=mode))

    def set_mask(self, pruning_perc):
        mask = weight_prune(self, pruning_perc)
        for i, m in enumerate(range(0, len(mask), 2)):
            self.recurrent_layers[i].set_mask((mask[m], mask[m+1]))

    def forward(self, input):
        batch_size = input.size(0) if self.batch_first else input.size(1)

        for l, hidden_size in enumerate(self.hidden_layers):
            if l == 0: out = input
            hx = torch.zeros(batch_size, hidden_size, dtype=input.dtype, device=input.device)
            out = self.layer_output(self.recurrent_layers[l], out, hx)

        if self.batch_first:
            return out[:, -1, :]
        return out[-1]

    def layer_output(self, layer, input, hx):
        dim = 1 if self.batch_first else 0
        n_seq = input.size(dim)
        outputs = []

        if self.mode == 'lstm':
            cx = copy.deepcopy(hx)

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]
            if self.mode == 'lstm':
                hx, cx = layer(seq, (hx, cx))
            else:
                hx = layer(seq, hx)
            output = hx
            outputs.append(output.unsqueeze(dim))

        return torch.cat(outputs, dim=dim)


class MaskedRecurrentLayer(nn.RNNCellBase):
    def __init__(self, input_size, hidden_size, mode='tanh'):
        self.mode = mode
        self.masked = False
        if mode == 'tanh' or mode == 'relu':
            num_chunks = 1
        elif mode == 'gru':
            num_chunks = 3
        elif mode == 'lstm':
            num_chunks = 4
        super(MaskedRecurrentLayer, self).__init__(input_size, hidden_size, bias=True, num_chunks=num_chunks)
    
    def set_mask(self, mask):
        self.register_buffer('input_mask', mask[0])
        self.register_buffer('hidden_mask', mask[1])
        self.masked = True
    
    def get_mask(self):
        return Variable(self.input_mask), Variable(self.hidden_mask)
    
    def forward(self, input, hx):
        cells = {
            'tanh': _VF.rnn_tanh_cell,
            'relu': _VF.rnn_relu_cell,
            'lstm': _VF.lstm_cell,
            'gru': _VF.gru_cell
        }
        cell = cells[self.mode]
        if self.masked == True:
            in_mask, hid_mask = self.get_mask()
            weight_ih = self.weight_ih * in_mask
            weight_hh = self.weight_hh * hid_mask
            return cell(input, hx, weight_ih, weight_hh, self.bias_ih, self.bias_hh)
        else:
            return cell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)