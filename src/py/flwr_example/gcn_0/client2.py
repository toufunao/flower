import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from load_cora import *
import sys
import flwr as fl
from collections import OrderedDict
import time
sys.path.append("cora")
log_info = []


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
    
        Args: 
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        support = torch.mm(input_feature, self.weight.to(device))
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias.to(device)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# ## 模型定义
class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


# ## 模型训练

# 超参数定义
learning_rate = 0.01
weight_decay = 5e-4
epochs = 200

# 模型定义：Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GcnNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

adjacency, features, labels, train_mask, val_mask, test_mask = load_data(path="split_data/cora2/")
tensor_x = features.to(device)  # X
tensor_y = labels.to(device)  # Y
tensor_train_mask = torch.from_numpy(train_mask).to(device)
tensor_val_mask = torch.from_numpy(val_mask).to(device)
tensor_test_mask = torch.from_numpy(test_mask).to(device)
indices = torch.from_numpy(np.asarray([adjacency.row, adjacency.col]).astype('int64')).long()
values = torch.from_numpy(adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (len(features), len(features))).to(device)


# 训练主体函数
def train(model, learning_rate, weight_decay, tensor_x, tensor_y, tensor_train_mask):
    criterion = nn.CrossEntropyLoss().to(device)  # 放train里
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 放train里
    # loss_history = []
    # val_acc_history = []
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)  # 前向传播
        train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)  # 计算损失值
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
    #     train_acc, _, _ = test(tensor_train_mask)     # 计算当前模型训练集上的准确率
    #     val_acc, _, _ = test(tensor_val_mask)     # 计算当前模型在验证集上的准确率
    #     # 记录训练过程中损失值和准确率的变化，用于画图
    #     loss_history.append(loss.item())
    #     val_acc_history.append(val_acc.item())
    #     print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
    #         epoch, loss.item(), train_acc.item(), val_acc.item()))

    # return loss_history, val_acc_history


# 测试函数
def test(model, tensor_val_mask):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)

        val_mask_logits = logits[tensor_val_mask]
        predict_y = val_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[tensor_val_mask]).float().mean()
        loss += criterion(val_mask_logits, tensor_y[tensor_val_mask]).item()
        log_info.append([loss, float(accuarcy)])
        print('测试', loss, float(accuarcy))
    return loss, accuarcy


class GCNClient(fl.client.NumPyClient):
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
        train(model, learning_rate, weight_decay, tensor_x, tensor_y, tensor_train_mask)
        cp = random.randint(0, 10)
        time.sleep(cp)
        t = time.time() - t
        return self.get_parameters(), len(train_mask), {"fit_time": float(t)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        t = time.time()
        loss, accuracy = test(model, tensor_test_mask)
        return loss, len(test_mask), {"accuracy": float(accuracy), "eval_time": float(time.time() - t)}


from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="GCN Client")
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
    fl.client.start_numpy_client(args.server_address, client=GCNClient())
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
