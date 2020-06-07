import random
import statistics
import sys
from timeit import default_timer as timer
import argparse

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
from pypaddle.sparse import LayeredGraph, CachedLayeredGraph
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')
from sparse import ArbitraryStructureRNN

BATCH_SIZE = 128
INPUT_SIZE = 128
EMBEDDING_DIM = 100
OUTPUT_SIZE = 2
EPOCHS = 15
RESULT_FILE_PATH = 'results/structure/'
STATE_DICT_PATH = 'state_dicts/structure/'


def create_variable(tensor):
    return Variable(tensor.to(device))


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
    def __init__(self, input_size, output_size, structure: LayeredGraph, mode):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=structure.first_layer_size)
        self.recurrent = ArbitraryStructureRNN(input_size=structure.first_layer_size, structure=structure, mode=mode)
        self.out = nn.Linear(structure.last_layer_size, output_size)

    def forward(self, input):
        input = input.t()
        embedded = self.embedding(input)
        recurrent_output = self.recurrent(embedded)
        return self.out(recurrent_output)


def train(model, epochs, train_loader, test_loader, criterion, optimizer, mode):
    for epoch in range(epochs):
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
        test_acc, test_loss = test(model, test_loader, criterion)

    return test_acc, test_loss


def test(model, test_loader, criterion):
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

        test_acc = correct / total
        return test_acc, test_loss


def get_graph_properties(graph):
    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)

    eccentricity = nx.eccentricity(graph)
    eccentricity_mean = statistics.mean(eccentricity.values())
    eccentricity_var = statistics.variance(eccentricity.values())
    eccentricity_std = statistics.stdev(eccentricity.values())

    diameter = nx.diameter(graph, eccentricity)
    density = num_edges / (num_nodes * (num_nodes - 1))

    degree = nx.degree_centrality(graph)
    degree_mean = statistics.mean(degree.values())
    degree_var = statistics.variance(degree.values())
    degree_std = statistics.stdev(degree.values())

    closeness = nx.closeness_centrality(graph)
    closeness_mean = statistics.mean(closeness.values())
    closeness_var = statistics.variance(closeness.values())
    closeness_std = statistics.stdev(closeness.values())

    nodes_betweenness = nx.betweenness_centrality(graph)
    nodes_betweenness_mean = statistics.mean(nodes_betweenness.values())
    nodes_betweenness_var = statistics.variance(nodes_betweenness.values())
    nodes_betweenness_std = statistics.stdev(nodes_betweenness.values())

    edge_betweenness = nx.edge_betweenness_centrality(graph)
    edge_betweenness_mean = statistics.mean(edge_betweenness.values())
    edge_betweenness_var = statistics.variance(edge_betweenness.values())
    edge_betweenness_std = statistics.stdev(edge_betweenness.values())

    return num_nodes, num_edges, diameter, density, \
           eccentricity_mean, eccentricity_var, eccentricity_std, \
           degree_mean, degree_var, degree_std, \
           closeness_mean, closeness_var, closeness_std, \
           nodes_betweenness_mean, nodes_betweenness_var, nodes_betweenness_std, \
           edge_betweenness_mean, edge_betweenness_var, edge_betweenness_std


def main(random_graph, graph_name, graph_nr, mode):
    run_start = timer()

    random_structure = CachedLayeredGraph()
    random_structure.add_edges_from(random_graph.edges)
    random_structure.add_nodes_from(random_graph.nodes)

    num_layers = len(random_structure.layers)
    num_nodes, num_edges, diameter, density, \
    eccentricity_mean, eccentricity_var, eccentricity_std, \
    degree_mean, degree_var, degree_std, \
    closeness_mean, closeness_var, closeness_std, \
    nodes_betweenness_mean, nodes_betweenness_var, nodes_betweenness_std, \
    edge_betweenness_mean, edge_betweenness_var, edge_betweenness_std = get_graph_properties(random_graph)

    source_nodes, sink_nodes = random_structure.first_layer_size, random_structure.last_layer_size

    train_loader, test_loader = get_reber_loaders(BATCH_SIZE)

    model = Model(INPUT_SIZE, OUTPUT_SIZE, random_structure, mode=mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    test_acc, test_loss = train(model, EPOCHS, train_loader, test_loader, criterion, optimizer, mode)

    run_end = timer()
    total_run = run_end - run_start

    print('[{}] · {}_{} · [Testing] Loss: {:7.3f}, Acc: {:.3f} · [Time] {:6.2f} s'.format(mode, graph_name, graph_nr, test_loss, test_acc, total_run))
    f = open(RESULT_FILE_PATH + '{}.csv'.format(mode.lower()), 'a')
    f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(mode, graph_name, graph_nr, num_layers, num_nodes, num_edges,
                                                                                                           source_nodes, sink_nodes, diameter, density,
                                                                                                           eccentricity_mean, eccentricity_var, eccentricity_std,
                                                                                                           degree_mean, degree_var, degree_std,
                                                                                                           closeness_mean, closeness_var, closeness_std,
                                                                                                           nodes_betweenness_mean, nodes_betweenness_var, nodes_betweenness_std,
                                                                                                           edge_betweenness_mean, edge_betweenness_var, edge_betweenness_std,
                                                                                                           test_acc, test_loss, total_run))
    f.close()
    torch.save(model.state_dict(), STATE_DICT_PATH + '{}/{}_{}.pt'.format(mode, graph_name, graph_nr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arbitrary Structure Experiment')
    parser.add_argument('--mode', default='RNN_TANH', type=str, help='Available modes: RNN_TANH, RNN_RELU, GRU, LSTM')
    parser.add_argument('--gpu_device', default=0, type=int, help='GPU device index')
    args = parser.parse_args()

    mode = args.mode
    device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')

    print('--- Mode: {} ---'.format(mode))

    f = open(RESULT_FILE_PATH + '{}.csv'.format(mode.lower()), 'w')
    f.write('mode,graph,graph_nr,layers,nodes,edges,source_nodes,sink_nodes,diameter,density,eccentricity_mean,eccentricity_var,eccentricity_std,'
            'degree_mean,degree_var,degree_std,closeness_mean,closeness_var,closeness_std,'
            'nodes_betweenness_mean,nodes_betweenness_var,nodes_betweenness_std,'
            'edge_betweenness_mean,edge_betweenness_var,edge_betweenness_std,test_acc,test_loss,time\n')
    f.close()

    for g in range(100):
        n = random.randrange(10, 51)
        random_graph = nx.barabasi_albert_graph(n, n // 10)
        main(random_graph, 'barabasi_albert', g, mode)

    for g in range(100):
        n = random.randrange(10, 51)
        random_graph = nx.connected_watts_strogatz_graph(n, n // 5, 1, 10)
        main(random_graph, 'watts_strogatz', g, mode)
