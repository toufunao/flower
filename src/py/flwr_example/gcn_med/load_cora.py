import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
import random


def normalize_adj(adjacency):
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


def normalize_features(features):
    return features / features.sum(1)


def split(full_list, shuffle=False, ratio_1=0.2, ratio_2=0.4):
    n_total = len(full_list)
    offset_1 = int(n_total * ratio_1)
    offset_2 = int(n_total * (ratio_1 + ratio_2))
    if n_total == 0 or offset_1 < 1 or offset_2 < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset_1]
    sublist_2 = full_list[offset_1:offset_2]
    sublist_3 = full_list[offset_2:]
    return sublist_1, sublist_2, sublist_3


def load_data(path, dataset="pubmed"):
    """Load pubmed dataset for now"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    encode_label = LabelEncoder()
    labels = encode_label.fit_transform(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    old_edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges_unordered = []
    for x in old_edges_unordered:
        if x[0] in idx:
            if x[1] in idx:
                edges_unordered.append(list(x))
    edges_unordered = np.array(edges_unordered)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    features = normalize_features(features)
    adj = normalize_adj(adj)

    node_list = [idx_map[item[0].astype(np.int32)] for item in idx_features_labels]
    idx_train, idx_val, idx_test = split(node_list, shuffle=False, ratio_1=0.4, ratio_2=0.0)
    features = torch.FloatTensor(np.array(features))
    labels = torch.tensor(np.array(labels))
    num_nodes = features.shape[0]
    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return adj, features, labels, train_mask, val_mask, test_mask
