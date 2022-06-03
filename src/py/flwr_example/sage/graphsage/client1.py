import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator

import flwr as fl
from collections import OrderedDict

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("../cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


np.random.seed(1)
random.seed(1)
num_nodes = 2708
feat_data, labels, adj_lists = load_cora()
features = nn.Embedding(2708, 1433)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

agg1 = MeanAggregator(features, cuda=True)
enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
               base_model=enc1, gcn=True, cuda=False)
enc1.num_samples = 5
enc2.num_samples = 5

graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
rand_indices = np.random.permutation(num_nodes)
test = rand_indices[:1000]
val = rand_indices[1000:1500]
train = list(rand_indices[1500:])

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)


def f_train(graphsage, train):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    for batch in range(100):
        times = []
        batch_nodes = train[:256]  # 训练误差
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        # print(batch,loss.data.item())


def f_test(graphsage, val):
    batch_nodes = val[:256]  # 验证误差
    val_output = graphsage.forward(val)
    loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    acc = accuracy_score(labels[val], val_output.data.numpy().argmax(axis=1))
    print('测试', loss.data.item(), acc)
    return loss.data.item(), acc


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in graphsage.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(graphsage.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        graphsage.load_state_dict(state_dict, strict=True)

    def get_properties(self, config):
        pass

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        f_train(graphsage, train)
        return self.get_parameters(), len(train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = f_test(graphsage, test)
        return loss, len(test), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080",
        help=f"gRPC server address (default: '[::]:8080')",
    )
    args = parser.parse_args()
    fl.client.start_numpy_client(args.server_address, client=CifarClient())
