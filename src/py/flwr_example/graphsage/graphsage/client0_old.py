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
from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from sklearn.metrics import accuracy_score
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

        scores = torch.where(torch.isnan(scores), torch.full_like(scores, 0), scores)  + 1e-10
        # print(scores)
        return self.xent(scores, labels.squeeze())
        

def normalize_features(features):
    return features / features.sum(1)


def load_cora():
    num_nodes = len(pd.read_table("cora0/cora.content",header=None))
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora0/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(np.float64, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
    
    adj_lists = defaultdict(set)
    with open("cora0/cora.cites") as fp:
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
print(num_nodes)
features = nn.Embedding(num_nodes, 1433)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

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
test = rand_indices[:int(0.4*num_nodes)]
val = rand_indices[int(0.4*num_nodes):int(0.6*num_nodes)]
train = list(rand_indices[int(0.6*num_nodes):])

    # # for round in range(10):
    # for batch in range(100):
    #     batch_nodes = train[:25]
    #     random.shuffle(train)
    #     optimizer.zero_grad()
    #     loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    #     loss.backward()
    #     print(batch, loss.data)


    # val_output = graphsage.forward(val)
    # print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))

def f_train(graphsage,train,labels):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.1)
    for batch in range(100):
        batch_nodes = train[:50]  #训练误差
        random.shuffle(train)
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        # print(loss)
        loss.backward()
        optimizer.step()
        # print(batch,loss.data.item())

def f_test(graphsage,val,labels):
    batch_nodes = val[:50]  #验证误差
    val_output = graphsage.forward(val)
    loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    acc = accuracy_score(labels[val], val_output.data.numpy().argmax(axis=1))
    print('测试',loss.data.item(),acc)
    return loss.data.item(),acc

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in graphsage.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(graphsage.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        graphsage.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        f_train(graphsage,train,labels)
        return self.get_parameters(), len(train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss,accuracy= f_test(graphsage,test,labels)
        
        return loss, len(test), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())
