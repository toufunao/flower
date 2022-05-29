import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelBinarizer

def normalize_adj(adjacency):
  adjacency += sp.eye(adjacency.shape[0])
  degree = np.array(adjacency.sum(1))
  d_hat = sp.diags(np.power(degree, -0.5).flatten())
  return d_hat.dot(adjacency).dot(d_hat).tocoo()

def normalize_features(features):
  return features / features.sum(1)

def load_data(path="cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path,dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    encode_onehot = LabelBinarizer()
    labels = encode_onehot.fit_transform(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)


    features = normalize_features(features)
    adj = normalize_adj(adj)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.where(labels)[1])

    num_nodes = features.shape[0]
    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return adj, features, labels, train_mask, val_mask, test_mask