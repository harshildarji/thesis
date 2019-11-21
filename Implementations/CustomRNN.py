import torch
import torch.nn as nn

class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_layers : list, batch_first=False, nonlinearity='tanh'):
        super(DeepRNN, self).__init__()
        assert len(hidden_layers) > 0

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.batch_first = batch_first

        self.cell_in = nn.RNNCell(input_size, hidden_layers[0], 'relu')
        self.cells = [nn.RNNCell(hidden_layers[in_size], out_size) for in_size, out_size in enumerate(hidden_layers[1:])]
        
    def forward(self, input):
        batch_size = input.size(0) if self.batch_first else input.size(1)

        hx = torch.zeros(batch_size, self.hidden_layers[0], dtype=input.dtype, device=input.device)
        out = self.__cell(self.cell_in, input, hx)
        for i, hidden_size in enumerate(self.hidden_layers[1:]):
            hx = torch.zeros(batch_size, hidden_size, dtype=input.dtype, device=input.device)
            out = self.__cell(self.cells[i], out, hx)
            
        return out
            
    def __cell(self, cell, input, hx):
        dim = 1 if self.batch_first else 0
        n_seq = input.size(dim)
        outputs = []

        for i in range(n_seq):
            seq = input[:, i, :] if self.batch_first else input[i]
            hx = cell(seq, hx)
            outputs.append(hx.unsqueeze(dim))

        return torch.cat(outputs, dim=dim)