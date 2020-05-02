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
