from pyexpat import model
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
import random
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from load_data import *
import sys
import flwr as fl
from collections import OrderedDict
from torch.nn import Linear, Dropout
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv
import torch.nn.functional as F
import time

log_info = []
edge_index, x, y, train_mask, val_mask, test_mask = load_data("cora10/", dataset="cora")
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# Create batches with neighbor sampling
train_loader = NeighborLoader(
    data,
    num_neighbors=[3, 3],
    batch_size=5,
    input_nodes=data.train_mask,
)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, dim_in, dim_h, dim_out, lr):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer

        self.train()
        for epoch in range(epochs + 1):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            # Train on batches
            for batch in train_loader:
                optimizer.zero_grad()
                _, out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss
                acc += accuracy(out[batch.train_mask].argmax(dim=1),
                                batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()

                # Validation
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_acc += accuracy(out[batch.val_mask].argmax(dim=1),
                                    batch.y[batch.val_mask])


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    loss = criterion(out[data.test_mask], data.y[data.test_mask])
    print(acc, loss.item())
    return acc, loss.item()


# Initialize the model
epoch = 1
lr = 0.9
model = GraphSAGE(1433, 16, 7, lr)


class SageClient(fl.client.NumPyClient):
    def __init__(self, epochs):
        super(SageClient, self).__init__()
        self.epochs = epochs

    def get_properties(self, config):
        pass

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        t = time.time()
        model.fit(data, self.epochs)
        cp = random.randint(0, 10)
        time.sleep(cp)
        t = time.time() - t
        return self.get_parameters(), len(train_mask), {"fit_time": float(t), "rnd": config["rnd"]}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(model, data)
        return loss, len(test_mask), {"accuracy": float(accuracy)}


from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Sage Client")
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
    parser.add_argument(
        "--e",
        type=int,
        default=10,
        help=f"Training number. Default to 0",
    )
    args = parser.parse_args()
    fl.client.start_numpy_client(args.server_address, client=SageClient(args.e))
    import os

    if not os.path.exists('log/'):
        os.mkdir('log/')
    with open(f'log/client0_{args.n}.log', mode='w', encoding='utf-8') as f:
        f.write("[ ")
        for i in range(len(log_info)):
            item = log_info[i]
            f.write(" (" + str(i) + "," + str(item[0]) + "," + str(item[1]) + "),")
        f.write("]")
    print(f'training completed')
