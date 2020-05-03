import sys
from timeit import default_timer as timer

import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')
from sparse import PruneRNN

BATCH_SIZE = 32
INPUT_SIZE = 128
EMBEDDING_DIM = 100
OUTPUT_SIZE = 2
EPOCHS = 5
HIDDEN_LAYERS = [50, 50, 50]
MODES = ['RNN_TANH', 'RNN_RELU', 'GRU', 'LSTM']
FILE = 'results/prune_results.csv'
STATE_DICT_PATH = 'state_dicts/prune/'


def create_variable(tensor):
    return Variable(tensor)


def str2ascii(string):
    ascii_arr = [ord(s) for s in string]
    return ascii_arr, len(ascii_arr)


def pad_seq(vect_seqs, seq_lens, valid):
    seq_tensor = torch.zeros((len(vect_seqs), seq_lens.max())).long()

    for index, (seq, seq_len) in enumerate(zip(vect_seqs, seq_lens)):
        seq_tensor[index, :seq_len] = torch.LongTensor(seq)

    return create_variable(seq_tensor), create_variable(valid)


def make_variables(strings, valid):
    seqs_and_lens = [str2ascii(string) for string in strings]
    vect_seqs = [s[0] for s in seqs_and_lens]
    seq_lens = torch.LongTensor([s[1] for s in seqs_and_lens])
    valid = torch.LongTensor(valid)
    return pad_seq(vect_seqs, seq_lens, valid)


class MakeDataset(Dataset):
    def __init__(self, data):
        self.strings = list(data['string'])
        self.valid = list(data['valid'])
        self.len = len(self.valid)
        self.valid_list = [0, 1]

    def __getitem__(self, index):
        return self.strings[index], self.valid[index]

    def __len__(self):
        return self.len


def get_reber_loaders(batch_size):
    train_data = pd.read_csv('../dataset/train_data.csv')
    test_data = pd.read_csv('../dataset/test_data.csv')
    train = MakeDataset(train_data)
    test = MakeDataset(test_data)
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers: list, mode):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_layers[0])
        self.recurrent = PruneRNN(hidden_layers[0], hidden_layers, mode=mode)
        self.out = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, input):
        input = input.t()
        embedded = self.embedding(input)
        recurrent_output = self.recurrent(embedded)
        return self.out(recurrent_output)


def train(model, epochs, train_loader, test_loader, criterion, optimizer, mode):
    for epoch in range(epochs):
        start = timer()
        model.train()

        train_loss = 0
        correct = 0
        total = 0

        for i, (string, valid) in enumerate(train_loader):
            input, target = make_variables(string, valid)
            output = model(input)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.data.item()
            _, predict = torch.max(output.data, 1)
            total += target.size(0)
            correct += predict.eq(target.data).cpu().sum().item()

        train_acc = correct / total
        test(model, test_loader, criterion, start, mode, epoch, train_loss, train_acc)


def test(model, test_loader, criterion, start, mode, epoch, train_loss, train_acc):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (string, valid) in enumerate(test_loader):
            input, target = make_variables(string, valid)
            output = model(input)

            loss = criterion(output, target)
            test_loss += loss.item()
            _, predict = torch.max(output.data, 1)
            total += target.size(0)
            correct += predict.eq(target.data).cpu().sum().item()

        end = timer()

        test_acc = correct / total
        time = end - start

        print('[{}] · Epoch {:2d} · [Training] Loss: {:7.3f}, Acc: {:.3f} · [Testing] Loss: {:7.3f}, Acc: {:.3f} · [Time] {:6.2f} s'.format(mode, epoch + 1, train_loss, train_acc, test_loss, test_acc, time))

        f = open(FILE, 'a')
        f.write('{},{},{},{},{},{},{}\n'.format(mode, epoch + 1, train_loss, train_acc, test_loss, test_acc, time))
        f.close()


def get_model(mode):
    model = Model(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS, mode=mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


if __name__ == '__main__':
    f = open(FILE, 'w')
    f.write('mode,epoch,train_loss,train_acc,test_loss,test_acc,time\n')
    f.close()

    train_loader, test_loader = get_reber_loaders(BATCH_SIZE)

    for mode in MODES:
        print('--- Mode: {} ---'.format(mode))
        model, optimizer, criterion = get_model(mode)
        train(model, EPOCHS, train_loader, test_loader, criterion, optimizer, mode)
        torch.save(model.state_dict(), STATE_DICT_PATH + '{}.pt'.format(mode.lower()))
        print()
