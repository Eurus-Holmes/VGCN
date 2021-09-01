import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import Parameter
from tools.model_utils import get_laplace_mat

class GIN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 layer_num=2,
                 mid_channels=256,
                 dropout=0.6,
                 momentum=0.1,
                 add_self_loop=False
                 ):
        super(GIN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.dropout = dropout
        self.layer_num = layer_num
        self.momentum = momentum
        self.add_self_loop = add_self_loop

        self.eps = nn.Parameter(torch.zeros(self.layer_num))
        self.layer_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for idx in range(self.layer_num):
            in_size = mid_channels
            out_size = mid_channels
            if idx == 0:
                in_size = in_channels
            if idx == self.layer_num - 1:
                out_size = out_channels
            linear_layer = nn.Linear(in_size, out_size)
            self.layer_list.append(linear_layer)
            bn = nn.BatchNorm1d(out_size, momentum=self.momentum, track_running_stats=True)
            self.bn_list.append(bn)

        self.mlp_activation = nn.ELU()

    def forward(self, node_state, adj_mat):
        # h_{v}^{(k)}=\operatorname{MLP}^{(k)}\left(\left(1+\epsilon^{(k)}\right) \cdot h_{v}^{(k-1)}+\sum_{u \in \mathcal{N}(v)} h_{u}^{(k-1)}\right)
        h = node_state
        for idx in range(self.layer_num):
            h_hat = self.eps[idx] * h + torch.mm(adj_mat, h)
            h = self.layer_list[idx](h_hat)
            if idx != self.layer_num - 1:
                h = self.bn_list[idx](h)
                h = self.mlp_activation(h)
        return h


class VirtualGIN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=256,
                 layer_num=5,
                 dropout=0.6,
                 bias=False,
                 momentum=0.1,
                 ):
        super(VirtualGIN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.dropout = dropout
        self.bias = bias  # gcn
        self.layer_num = layer_num
        self.momentum = momentum

        self.gin_model = GIN(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             mid_channels=self.mid_channels,
                             layer_num=self.layer_num,
                             momentum=self.momentum
                             )

    def forward(self, node_ft, adj):
        out = self.gin_model(node_ft, adj)
        return out


from dataset.load_dataset import GlobalFlu

import torch
import torch.nn.functional as F
import numpy as np
import random
from tools.evaluate_utils import evaluate_regression


class VirtualGINTrainer(object):
    def __init__(self, wind_size=52, pred_step=1, layer_num=2, data_type='us', split_param=[0.6, 0.2, 0.2], seed=3):
        self.setup_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('FULL VGIN')
        self.dataset = GlobalFlu(wind_size=wind_size, pred_step=pred_step, data_type=data_type, split_param=split_param)
        self.dataset.to_tensor()
        self.dataset.to_device(self.device)
        self.adj = torch.full([self.dataset.node_size, self.dataset.node_size], 1, dtype=torch.float32).to(self.device)

        self.epochs = 200
        self.count = 0
        self.loss_type = 'mse'
        self.alpha = 0.00005
        self.layer_num = layer_num
        self.pred_nums = None
        self.min_loss = 1e10
        self.batch_size = 30
        self.build_model()

    def build_model(self):
        self.model = VirtualGIN(
            in_channels=self.dataset.node_feature_size,
            out_channels=self.dataset.label_size,
            layer_num=self.layer_num,
            momentum=0.8
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.best_res = 0

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
        for idx in range(train_size // self.batch_size):
            self.optimizer.zero_grad()
            right_bound = min((idx + 1) * self.batch_size, train_size + 1)
            shuffle_batch_idx = shuffle_idx[idx * self.batch_size: right_bound]
            batch_idx = self.dataset.train_index[shuffle_batch_idx]
            batch_node_label = self.dataset.label_mat[batch_idx]

            out = self.model(self.dataset.ft_mat, self.adj)
            out = out[batch_idx]

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

        # print(pred.shape, label.shape)
        train_mse, train_mae, train_mape = evaluate_regression(
            pred[self.dataset.train_index], label[self.dataset.train_index])

        valid_mse, valid_mae, valid_mape = evaluate_regression(
            pred[self.dataset.valid_index], label[self.dataset.valid_index])

        test_mse, test_mae, test_mape = evaluate_regression(
            pred[self.dataset.test_index], label[self.dataset.test_index])


        return train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
               train_mape, valid_mape, test_mape, pred

    def mape_loss(self, pred, label):
        errors = torch.abs((pred - label) / label)
        errors = errors / label.size()[0]
        loss = torch.sum(errors)
        return loss

    def mae_loss(self, pred, label):
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

    def start(self, display=True):
        self.test_acc_list = []
        for epoch in range(0, self.epochs):
            self.train()
            train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
            train_mape, valid_mape, test_mape, pred = self.test()

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
  for seed in [3]:
    mse_res_list = []
    mape_res_list = []
    print('seed = ', seed)
    for pred_step in [1, 3, 6]:
        for data_type in ['us']:
            for wind_size in [6, 9, 12]:
                res = VirtualGINTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type, seed=seed, layer_num=4).start(display=True)
                mse_res_list.append(res[0])  # mse
                mape_res_list.append(res[2])  # mape



