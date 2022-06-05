import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
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


def load_data(path, path_all="cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # 全局的编码方式
    idx_features_labels = np.genfromtxt("{}{}.content".format(path_all, dataset),
                                        dtype=np.dtype(str))  # 全局结点，主要看embedding对应关系
    encode_label = LabelEncoder()
    all_labels = encode_label.fit_transform(idx_features_labels[:, -1])

    # 子图的编码方式
    idx_sub_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))  # 子结点，用全局信息进行embedding

    features = []
    for x in idx_features_labels:
        if x[0] in idx_sub_features_labels[:, 0]:
            features.append(x[1:-1])

        else:
            features.append([100 for i in range(1433)])
    features = sp.csr_matrix(np.array(features), dtype=np.float32)

    all_label = idx_features_labels[:, -1]
    labels = []  # 根据整图得到子图的label的one-hot编码
    for i in range(len(idx_features_labels)):
        if idx_features_labels[i][0] in idx_sub_features_labels[:, 0]:
            labels.append(all_labels[i])
        else:
            labels.append(100)
    # print(labels)
    # labels = []   # 根据整图得到子图的label的one-hot编码
    # for i in range(len(idx_features_labels)):
    #   if idx_features_labels[i][0] in idx_sub_features_labels[:,0]:
    #     labels.append(all_labels[i])
    #   else:
    #      labels.append([0 for i in range(7)])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 整图的点label
    idx_map = {j: i for i, j in enumerate(idx)}  # 整图的map方式
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 子图的边的信息
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)  # 用整图的embedding看子图的边
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(all_labels.shape[0], all_labels.shape[0]), dtype=np.float32)

    features = normalize_features(features)
    adj = normalize_adj(adj)

    node_list = [idx_map[item[0].astype(np.int32)] for item in idx_sub_features_labels]
    idx_train, idx_val, idx_test = split(node_list, shuffle=False, ratio_1=0.2, ratio_2=0.4)
    features = torch.FloatTensor(np.array(features))
    labels = torch.tensor(np.array(labels))
    # print(labels.shape)
    num_nodes = features.shape[0]
    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return adj, features, labels, train_mask, val_mask, test_mask
