import torch
import torch.nn as nn
import copy

class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_layers : list, batch_first=False, mode='tanh', theta: float=None):
        super(DeepRNN, self).__init__()
        assert len(hidden_layers) > 0

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.batch_first = batch_first
        self.mode = mode.lower()
        self.theta = theta

        if self.mode == 'gru':
            self.cell_in = nn.GRUCell(input_size, hidden_layers[0])
            self.cells = nn.ModuleList([nn.GRUCell(hidden_layers[in_size], out_size) for in_size, out_size in enumerate(hidden_layers[1:])])
        elif self.mode == 'lstm':
            self.cell_in = nn.LSTMCell(input_size, hidden_layers[0])
            self.cells = nn.ModuleList([nn.LSTMCell(hidden_layers[in_size], out_size) for in_size, out_size in enumerate(hidden_layers[1:])])
        elif self.mode == 'tanh' or self.mode == 'relu':
            self.cell_in = nn.RNNCell(input_size, hidden_layers[0], nonlinearity=self.mode)
            self.cells = nn.ModuleList([nn.RNNCell(hidden_layers[in_size], out_size, nonlinearity=self.mode) for in_size, out_size in enumerate(hidden_layers[1:])])
        else:
            raise ValueError("Unknown value for mode. Expected 'LSTM', 'GRU', 'TANH' or 'RELU', got '{}'.".format(mode))
        
    def forward(self, input):
        hiddens = []
        batch_size = input.size(0) if self.batch_first else input.size(1)

        hx = torch.zeros(batch_size, self.hidden_layers[0], dtype=input.dtype, device=input.device)
        out = self.__cell__(self.cell_in, input, hx)
        hiddens.append(out)
        for i, hidden_size in enumerate(self.hidden_layers[1:]):
            hx = torch.zeros(batch_size, hidden_size, dtype=input.dtype, device=input.device)
            out = self.__cell__(self.cells[i], out, hx)
            hiddens.append(out)
            
        return out, hiddens
            
    def __cell__(self, cell, input, hx):
        dim = 1 if self.batch_first else 0
        n_seq = input.size(dim)
        outputs = []

        if self.mode == 'lstm':
            cx = copy.deepcopy(hx)

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]
            if self.mode == 'lstm':
                hx, cx = cell(seq, (hx, cx))
            else:
                hx = cell(seq, hx)
            hx = self.__mask__(hx)
            output = cx if self.mode == 'lstm' else hx
            outputs.append(output.unsqueeze(dim))

        return torch.cat(outputs, dim=dim)

    def __mask__(self, hx):
        if self.theta is None:
            return hx
        self.mask = torch.ones(hx.shape, dtype=torch.bool)
        self.mask[torch.where(abs(hx) < self.theta)] = False
        return hx * self.mask