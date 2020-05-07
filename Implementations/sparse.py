import numpy as np
import torch
import torch.nn as nn
from layer import MaskedRecurrentLayer
from pypaddle.sparse import LayeredGraph


class PruneRNN(nn.Module):
    def __init__(self, input_size, hidden_layers: list, mode='RNN_TANH', batch_first=False):
        super(PruneRNN, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.mode = mode.upper()
        self.batch_first = batch_first

        self.recurrent_layers = nn.ModuleList()
        for l, hidden_size in enumerate(hidden_layers):
            input_size = input_size if l == 0 else hidden_layers[l - 1]
            self.recurrent_layers.append(MaskedRecurrentLayer(input_size, hidden_size, mode, batch_first))

    def forward(self, input):
        batch_size = input.size(0) if self.batch_first else input.size(1)

        for l, hidden_size in enumerate(self.hidden_layers):
            hx = torch.zeros(batch_size, hidden_size, dtype=input.dtype, device=input.device)
            input = self.__step(self.recurrent_layers[l], input, hx)

        output = input[:, -1, :] if self.batch_first else input[-1]
        return output

    def __step(self, layer, input, hx):
        in_dim = 1 if self.batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []

        if self.mode == 'LSTM':
            cx = hx.clone()

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]

            if self.mode == 'LSTM':
                hx, cx = layer(seq, (hx, cx))
            else:
                hx = layer(seq, hx)

            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)

    def apply_mask(self, percent=0, i2h=False, h2h=False):
        """
        :param percent: Amount of pruning to apply. Default '0'
        :param i2h: If True, then Input-to-Hidden layers will be pruned. Default 'False'
        :param h2h: If True, then Hidden-to-Hidden layers will be pruned. Default 'False'
        :type percent: int
        :type i2h: bool
        :type h2h: bool
        """
        if not i2h and not h2h:
            return

        masks = self.__get_masks(percent, i2h, h2h)
        for l, layer in enumerate(self.recurrent_layers):
            if i2h:
                layer.set_i2h_mask(masks[l][0])
            if h2h:
                layer.set_h2h_mask(masks[l][-1])

    def __get_masks(self, percent, i2h, h2h):
        if i2h and h2h:
            key = ''
        elif i2h:
            key = 'ih'
        elif h2h:
            key = 'hh'

        weights = []
        for param, data in self.named_parameters():
            if 'bias' not in param and key in param:
                weights += list(data.cpu().data.abs().numpy().flatten())
        threshold = np.percentile(np.array(weights), percent)

        masks = {}
        for l, layer in enumerate(self.recurrent_layers):
            masks[l] = []
            for param, data in layer.named_parameters():
                if 'bias' not in param and key in param:
                    mask = torch.ones(data.shape, dtype=torch.bool, device=data.device)
                    mask[torch.where(abs(data) < threshold)] = False
                    masks[l].append(mask)

        return masks


class ArbitraryStructureRNN(nn.Module):
    def __init__(self, input_size, structure: LayeredGraph, mode='RNN_TANH', batch_first=False):
        super(ArbitraryStructureRNN, self).__init__()

        self.input_size = input_size
        self.mode = mode.upper()
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
            layer.set_i2h_mask(mask)

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
                    skip_layer.set_i2h_mask(mask)

                    skip_layers.append(skip_layer)
                    self._skip_targets[target_layer].append({'layer': skip_layer, 'source': distant_source_layer})
        self.skip_layers = nn.ModuleList(skip_layers)

    def forward(self, input):
        batch_size = input.size(0) if self.batch_first else input.size(1)

        layer_results = dict()
        for layer, layer_idx in zip(self.recurrent_layers, self._structure.layers):
            hx = torch.zeros(batch_size, self._structure.get_layer_size(layer_idx), dtype=input.dtype, device=input.device)
            input = self.__step(layer, input, hx)

            if layer_idx in self._skip_targets:
                for skip_target in self._skip_targets[layer_idx]:
                    source_layer = skip_target['layer']
                    source_idx = skip_target['source']

                    input += self.__step(source_layer, layer_results[source_idx], hx)

            layer_results[layer_idx] = input

        output = input[:, -1, :] if self.batch_first else input[-1]
        return output

    def __step(self, layer, input, hx):
        """
        This method is an exact copy of the '__step' method in 'PruneRNN'.

        Although it can be made common between both the classes, I decided
        to keep a separate copy for each class to avoid any trouble in case
        if, in the future, both classes are being defined in different files.
        """
        in_dim = 1 if self.batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []

        if self.mode == 'LSTM':
            cx = hx.clone()

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]

            if self.mode == 'LSTM':
                hx, cx = layer(seq, (hx, cx))
            else:
                hx = layer(seq, hx)

            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)
