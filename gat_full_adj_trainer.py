import torch.nn.functional as F


import os
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tools.model_utils import adj_2_bias,adj_2_bias_without_self_loop


class GatConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.6,
                 init_type='v1',
                 bias=True):
        super(GatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.bias = bias

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if self.bias and concat:
            self.parm_bias = Parameter(torch.Tensor(heads * out_channels))
        elif self.bias and not concat:
            self.parm_bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv_weight1 = Parameter(
            torch.Tensor(heads * out_channels, 1))

        self.conv_weight2 = Parameter(
            torch.Tensor(heads * out_channels, 1))

        nn.init.xavier_normal_(self.weight)
        if bias is True:
            nn.init.zeros_(self.parm_bias)
        nn.init.xavier_normal_(self.conv_weight1)
        nn.init.xavier_normal_(self.conv_weight2)

    def forward(self, nodes, adj_mat):
        adj_bias_mat = adj_2_bias(adj_mat)

        node_hidden_state = torch.mm(nodes, self.weight).view(-1, self.heads * self.out_channels)
        f1 = torch.mm(node_hidden_state, self.conv_weight1)
        f2 = torch.mm(node_hidden_state, self.conv_weight2)
        logits = f1 + torch.transpose(f2, 0, 1)
        if self.training and self.dropout > 0:
            logits = F.dropout(logits, p=0.6, training=True)
        coefs = F.softmax(F.leaky_relu(logits, negative_slope=self.negative_slope) + adj_bias_mat, dim=-1)

        if self.bias:
            vals = torch.mm(coefs, node_hidden_state) + self.parm_bias
        else:
            vals = torch.mm(coefs, node_hidden_state)

        return vals


class Model(nn.Module):
    def __init__(self,
                 node_num,
                 in_channels,
                 out_channels,
                 mlp_layer_num=2,
                 gat_layer_num=2,
                 mid_channels=256,
                 dropout=0.6,
                 bias=False
                 ):
        super(Model, self).__init__()
        self.node_num = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.bias = bias  # gat
        self.mlp_layer_num = mlp_layer_num
        self.gat_layer_num = gat_layer_num

        self.mlp_list = nn.ModuleList()
        self.gat_list = nn.ModuleList()

        for idx in range(self.mlp_layer_num):
            input_size = mid_channels
            if idx == 0:
                input_size = in_channels
            linear_layer = nn.Linear(input_size, mid_channels)
            self.mlp_list.append(linear_layer)

        for idx in range(self.gat_layer_num):
            out_size = mid_channels
            in_size = mid_channels
            if idx == 0:
                in_size = in_channels
            if idx == self.gat_layer_num - 1:
                out_size = out_channels
            gat_layer = GatConv(in_size, out_size, bias=self.bias)
            self.gat_list.append(gat_layer)

        self.activation = nn.ELU()

    def forward(self, node_ft, adj):
        for idx in range(self.gat_layer_num):
            node_ft = self.gat_list[idx](node_ft, adj)
            if idx != self.gat_layer_num - 1:
                node_ft = self.activation(node_ft)
                # node_ft = F.dropout(node_ft, p=self.dropout)
        out = node_ft
        return out


def get_adj(node_num, max_period=100):
    # adj = torch.eye(node_num, dtype=torch.long)
    adj = torch.eye(node_num, dtype=torch.float32)
    for i in range(node_num):
        for j in range(i):
            if i == j + 1:
                adj[i,j] = 1
            elif i > j and (i - j)%52==0:
                period = (i - j) / 52
                if max_period >= period:
                    adj[i, j] = 1
    return adj


from dataset.load_dataset import GlobalFlu
import torch
import torch.nn.functional as F
import numpy as np
import random
from tools.evaluate_utils import evaluate_regression


class GATFullTrainer(object):
    def __init__(self, wind_size=52, pred_step=1, layer=2, data_type='us', split_param=[0.6, 0.2, 0.2], seed=3):
        self.setup_seed(seed)
        print('GATFullTrainer  wind_size={}  pred_step={},  data_type={}'.format(wind_size, pred_step, data_type))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.dataset = GlobalFlu(wind_size=wind_size, pred_step=pred_step, data_type=data_type, split_param=split_param)

        self.dataset.to_tensor()
        self.dataset.to_device(self.device)

        self.adj = torch.full([self.dataset.node_size, self.dataset.node_size], 1, dtype=torch.float32).to(self.device)
        self.layer_num = layer
        self.epochs = 200
        self.count = 0
        self.loss_type = 'mse'

        self.pred_nums = None
        self.min_loss = 1e10
        self.batch_size = 30
        # self.batch_size = self.dataset.train_index.shape[0]

        self.build_model()

    def build_model(self):
        self.model = Model(
            node_num=self.dataset.node_size,
            in_channels=self.dataset.node_feature_size,
            out_channels=self.dataset.label_size,
            mlp_layer_num=2,
            gat_layer_num=2,
            dropout=0.3,
            bias=True
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.best_res = 0
        self.adj_mat = None


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(self):
        self.model.train()
        train_size = self.dataset.train_index.shape[0]
        shuffle_idx = torch.randperm(train_size)
        for idx in range( train_size // self.batch_size):
            self.optimizer.zero_grad()
            right_bound = min((idx + 1) * self.batch_size, train_size + 1)
            shuffle_batch_idx = shuffle_idx[idx * self.batch_size: right_bound]

            batch_idx = self.dataset.train_index[shuffle_batch_idx]
            batch_node_label = self.dataset.label_mat[batch_idx]

            out = self.model(self.dataset.ft_mat, self.adj)
            out = out[batch_idx]
            # print(batch_node_label.size(), out.size())
            # print(out)
            loss = 0
            if self.loss_type == 'mse':
                loss = F.mse_loss(out, batch_node_label, reduction='mean')
            if self.loss_type == 'mape':
                loss = self.mape_loss(out, batch_node_label)
            if self.loss_type == 'mae':
                loss = self.mae_loss(out, batch_node_label)

            loss.backward()
            self.optimizer.step()


    def test(self):
        self.model.eval()

        pred = self.model(self.dataset.ft_mat, self.adj)
        label = self.dataset.label_mat.to('cpu').detach().numpy()
        pred = pred.to('cpu').detach().numpy()


        train_mse, train_mae, train_mape = evaluate_regression(
            pred[self.dataset.train_index], label[self.dataset.train_index])

        valid_mse, valid_mae, valid_mape = evaluate_regression(
            pred[self.dataset.valid_index], label[self.dataset.valid_index])

        test_mse, test_mae, test_mape = evaluate_regression(
            pred[self.dataset.test_index], label[self.dataset.test_index])


        return train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
               train_mape, valid_mape, test_mape, pred, None



    def mape_loss(self, pred, label):
        pred = pred.view(-1)
        errors = torch.abs((pred - label) / label)
        errors = errors / label.size()[0]
        loss = torch.sum(errors)
        # print(pred.size(), label.size(), pred, label, loss)
        return loss

    def mae_loss(self, pred, label):
        pred = pred.view(-1)
        errors = torch.abs(pred - label)
        loss = torch.mean(errors)
        return loss

    def print_best_res(self):
        train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
        train_mape, valid_mape, test_mape, epoch = self.best_res
        msg_log = 'Epoch: {:03d}, MSE: {:.4f}, Val: {:.4f}, Test: {:.4f}, ' \
                  'MAE: {:.4f}, Val: {:.4f}, Test: {:.4f}, ' \
                  'MAPE: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
            epoch, train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
            train_mape, valid_mape, test_mape)
        print(msg_log)

    def start(self, topn=20, display=False):
        self.test_acc_list = []
        for epoch in range(1, self.epochs):
            self.train()
            # for i in self.model.parameters():
            #     print(i)
            train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
            train_mape, valid_mape, test_mape, pred, adj_mat = self.test()

            if valid_mse < self.min_loss:
                self.min_loss = valid_mse
                self.pred_nums = pred
                self.best_res = [train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
                                          train_mape, valid_mape, test_mape]
                self.best_res.append(epoch)
            if display:
                mse_log = 'Epoch: {:03d}, MES Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(mse_log.format(epoch, train_mse, valid_mse, test_mse))
                mae_log = 'Epoch: {:03d}, MAE Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(mae_log.format(epoch, train_mae, valid_mae, test_mae))
                mape_log = 'Epoch: {:03d}, MAPE Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(mape_log.format(epoch, train_mape, valid_mape, test_mape))

        self.print_best_res()
        train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
        train_mape, valid_mape, test_mape, epoch = self.best_res
        return test_mse, test_mae, test_mape


if __name__ == '__main__':
    res_list = []
    for pred_step in [3, 6]:
        for data_type in ['us']:
            for wind_size in [6, 9, 12]:
                res = GATFullTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type, seed=3, layer=5).start(display=False)
                res_list.append(res[0]) # mse

    for idx in range(len(res_list)):
        print(res_list[idx])
        if (idx+1) % 3 == 0:
            print()
        if (idx + 1) % 9 == 0:
            print()
