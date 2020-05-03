import numpy as np
import torch
import torch.nn as nn
from layer import MaskedRecurrentLayer


class PruneRNN(nn.Module):
    def __init__(self, input_size, hidden_layers: list, mode='tanh', batch_first=False):
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
