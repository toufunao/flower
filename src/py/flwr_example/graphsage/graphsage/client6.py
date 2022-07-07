from pickle import FALSE
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import pandas as pd
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
import scipy.sparse as sp
from encoders import Encoder
from aggregators import MeanAggregator
from sklearn.metrics import accuracy_score
import flwr as fl
from collections import OrderedDict

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
log_info = []


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
        scores = torch.where(torch.isnan(scores), torch.full_like(scores, 0), scores)
        return self.xent(scores, labels.squeeze())


def load_cora():
    num_feats = 1433
    feat_data = np.zeros((201, num_feats))
    labels = np.empty((201, 1), dtype=np.int64)
    node_map = {}
    label_map = {}

    with open("../split_cora/cora6/cora.content") as fp:
        j = 0
        for i, line in enumerate(fp):
            if j < 201:
                info = line.strip().split()
                feat_data[i, :] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]
                j = j + 1
            else:
                break

    adj_lists = defaultdict(set)
    with open("../split_cora/cora6/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            if info[0] in node_map.keys():
                if info[1] in node_map.keys():
                    paper1 = node_map[info[0]]
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


np.random.seed(1)
random.seed(1)
feat_data, labels, adj_lists = load_cora()
num_nodes = len(feat_data)
features = nn.Embedding(num_nodes, 1433)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
# features.cuda()

agg1 = MeanAggregator(features, cuda=True)
enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
               base_model=enc1, gcn=True, cuda=False)
enc1.num_samples = 4
enc2.num_samples = 4

graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
rand_indices = np.random.permutation(num_nodes)
test = rand_indices[:int(0.4 * num_nodes)]
val = rand_indices[int(0.4 * num_nodes):int(0.6 * num_nodes)]
train = list(rand_indices[int(0.6 * num_nodes):])


def f_train(graphsage, train, labels):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.3)
    # for round in range(10):
    for batch in range(1000):
        batch_nodes = train[:50]
        random.shuffle(train)
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()


def f_test(graphsage, val, labels):
    batch_nodes = val  # 验证误差
    val_output = graphsage.forward(val)
    loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    acc = accuracy_score(labels[val], val_output.data.numpy().argmax(axis=1))
    print('测试', loss.data.item(), float(acc))
    log_info.append((loss.data.item(), acc))
    return loss.data.item(), acc


class SageClient(fl.client.NumPyClient):
    def get_parameters(self):
        # print("参数size:", [val.cpu().numpy() for _, val in graphsage.state_dict().items()])
        return [val.cpu().numpy() for _, val in graphsage.state_dict().items()]

    def get_properties(self, config):
        pass

    def set_parameters(self, parameters):
        params_dict = zip(graphsage.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        graphsage.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        t = time.time()
        f_train(graphsage, train, labels)
        # print(time.time() - t)
        # print(self.get_parameters()[0].shape)
        # print(len(self.get_parameters()) * self.get_parameters()[0].shape[0] * self.get_parameters()[0].shape[1])
        t = time.time() - t
        return self.get_parameters(), len(train), {"fit_time": float(t), "rnd": config["rnd"]}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        t = time.time()
        loss, accuracy = f_test(graphsage, test, labels)

        return loss, len(test), {"accuracy": float(accuracy), "eval_time": float(time.time() - t)}


from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Graphsage Client")
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080",
        help=f"gRPC server address (default: '[::]:8080')",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help=f"Training number. Default to 0",
    )
    args = parser.parse_args()
    fl.client.start_numpy_client(args.server_address, client=SageClient())
    import os

    if not os.path.exists('log/'):
        os.mkdir('log/')
    with open(f'log/client0_{args.n}.log', mode='w', encoding='utf-8') as f:
        f.write("[ ")
        for i in range(len(log_info)):
            item = log_info[i]
            f.write(" (" + str(i+1) + "," + str(item[0]) + "," + str(item[1]) + "),")
        f.write("]")
    print(f'training completed')
