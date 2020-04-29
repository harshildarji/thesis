import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pypaddle.sparse import LayeredGraph
from torch.autograd import Variable
from torch.nn import init
from torch.nn.parameter import Parameter


class MaskedRecurrentLayer(nn.Module):
    def __init__(self, input_size, hidden_size, mode='TANH'):
        super(MaskedRecurrentLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.batch_first = False

        if self.mode == 'tanh':
            self.activation = torch.tanh
        elif self.mode == 'relu':
            self.activation = torch.relu
        else:
            raise ValueError("Unknown nonlinearity '{}", format(self.mode))

        self._flat_weights_names = []
        self._all_weights = []

        w_ih = Parameter(torch.Tensor(self.hidden_size, self.input_size))
        w_hh = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        b_ih = Parameter(torch.Tensor(self.hidden_size))
        b_hh = Parameter(torch.Tensor(self.hidden_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)
        param_names = ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']
        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)

        self._flat_weights_names.extend(param_names)
        self._all_weights.append(param_names)
        self._flat_weights = [getattr(self, weight) for weight in self._flat_weights_names]
        # self.flatten_parameters()
        self.reset_parameters()

        self.register_buffer('mask', torch.ones((self.hidden_size, self.input_size), dtype=torch.bool))

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
                    all_weights, 4, self.input_size, rnn.get_cudnn_mode(self.mode),
                    self.hidden_size, 1, self.batch_first, bool(False))

    def reset_parameters(self, keep_mask=False):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

        if hasattr(self, 'mask') and not keep_mask:
            self.mask = torch.ones(self.weight_ih.size(), dtype=torch.bool)

    def get_mask(self):
        return self.mask

    def set_mask(self, mask):
        self.mask = Variable(mask)

    def get_weight_count(self):
        return self.mask.sum()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, nonlinearity={mode}, batch_first={batch_first}'
        return s.format(**self.__dict__)

    def forward(self, input, hx):
        igate = F.linear(input, self.weight_ih * self.mask, self.bias_ih)
        hgate = F.linear(hx, self.weight_hh, self.bias_hh)
        return self.activation(igate + hgate)


class ArbitraryRNN(nn.Module):
    def __init__(self, input_size, structure: LayeredGraph, batch_first=False, mode='tanh'):
        super(ArbitraryRNN, self).__init__()
        self.batch_first = batch_first

        self._structure = structure
        assert structure.num_layers > 0

        self.recurrent_layers = nn.ModuleList()
        for l, layer in enumerate(structure.layers):
            input_size = input_size if l == 0 else structure.get_layer_size(l - 1)
            self.recurrent_layers.append(
                MaskedRecurrentLayer(input_size, structure.get_layer_size(l), mode=mode))

        for layer_idx, layer in zip(structure.layers[1:], self.recurrent_layers[1:]):
            mask = torch.zeros(structure.get_layer_size(layer_idx), structure.get_layer_size(layer_idx - 1))
            for source_idx, source_vertex in enumerate(structure.get_vertices(layer_idx - 1)):
                for target_idx, target_vertex in enumerate(structure.get_vertices(layer_idx)):
                    if structure.has_edge(source_vertex, target_vertex):
                        mask[target_idx][source_idx] = 1
            layer.set_mask(mask)

        skip_layers = []
        self._skip_targets = {}
        for target_layer in structure.layers[2:]:
            target_size = structure.get_layer_size(target_layer)
            for distant_source_layer in structure.layers[:target_layer - 1]:
                if structure.layer_connected(distant_source_layer, target_layer):
                    if target_layer not in self._skip_targets:
                        self._skip_targets[target_layer] = []

                    skip_layer = MaskedRecurrentLayer(structure.get_layer_size(distant_source_layer), target_size, mode=mode)
                    mask = torch.zeros(structure.get_layer_size(target_layer), structure.get_layer_size(distant_source_layer))
                    for source_idx, source_vertex in enumerate(structure.get_vertices(distant_source_layer)):
                        for target_idx, target_vertex in enumerate(structure.get_vertices(target_layer)):
                            if structure.has_edge(source_vertex, target_vertex):
                                mask[target_idx][source_idx] = 1
                    skip_layer.set_mask(mask)

                    skip_layers.append(skip_layer)
                    self._skip_targets[target_layer].append({'layer': skip_layer, 'source': distant_source_layer})
        self.skip_layers = nn.ModuleList(skip_layers)

    def forward(self, input):
        batch_size = input.size(0) if self.batch_first else input.size(1)

        layer_results = dict()
        for layer, layer_idx in zip(self.recurrent_layers, self._structure.layers):
            hx = torch.zeros(batch_size, self._structure.get_layer_size(layer_idx), dtype=input.dtype, device=input.device)
            input = self.step(layer, input, hx)

            if layer_idx in self._skip_targets:
                for skip_target in self._skip_targets[layer_idx]:
                    source_layer = skip_target['layer']
                    source_idx = skip_target['source']

                    input += self.step(source_layer, layer_results[source_idx], hx)

            layer_results[layer_idx] = input

        output = input[:, -1, :] if self.batch_first else input[-1]
        return output

    def step(self, layer, input, hx):
        in_dim = 1 if self.batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]
            hx = layer(seq, hx)
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)
