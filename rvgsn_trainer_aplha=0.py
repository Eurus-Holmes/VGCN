import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import Parameter
from tools.model_utils import get_laplace_mat

class GCNConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 dropout=0.6,
                 bias=True,
                 init_type='v1'
                 ):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.bias = bias
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels)
        )
        # nn.init.normal_(self.weight)
        # linear init
        if init_type == 'v1':
            # bound = (1/in_channels)**0.5
            bound = (1 / in_channels)
            nn.init.uniform_(self.weight, -bound, bound)
            if bias is True:
                self.bias = Parameter(torch.Tensor(out_channels))
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            nn.init.xavier_normal_(self.weight)
            if bias is True:
                self.bias = Parameter(torch.Tensor(out_channels))
                nn.init.zeros_(self.bias)
        # self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, node_state, adj_mat):
        adj_mat = get_laplace_mat(adj_mat, type='sym')
        node_state = torch.mm(adj_mat, node_state)
        # node_state = self.linear(node_state)
        node_state = torch.mm(node_state, self.weight)
        if self.bias is not None:
            node_state = node_state + self.bias
        return node_state


class Model(nn.Module):
    def __init__(self,
                 node_num,
                 in_channels,
                 out_channels,
                 gcn_layer_num=2,
                 mid_channels=256,
                 dropout=0.6,
                 bias=False
                 ):
        super(Model, self).__init__()
        self.node_num = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.dropout = dropout
        self.bias = bias  # gcn
        self.gcn_layer_num = gcn_layer_num

        self.gcn_list = nn.ModuleList()
        self.linear_in = nn.Linear(self.in_channels, self.mid_channels)
        self.linear_in2 = nn.Linear(self.mid_channels, self.mid_channels)
        self.linear_out = nn.Linear(self.mid_channels, self.out_channels)

        for idx in range(self.gcn_layer_num):
            gcn_layer = GCNConv(mid_channels, mid_channels, bias=self.bias)
            self.gcn_list.append(gcn_layer)

        self.adj_trans_layer = nn.Linear(mid_channels, mid_channels)
        self.activation = nn.ELU()

    def forward(self, node_ft):
        batch_size = node_ft.size()[0]

        node_ft = self.activation(self.linear_in(node_ft))
        node_ft = F.dropout(node_ft, p=self.dropout, training=self.training)
        node_ft = self.activation(self.linear_in2(node_ft))
        node_ft = F.dropout(node_ft, p=self.dropout, training=self.training)

        res_mat = torch.zeros(batch_size, self.mid_channels).to(node_ft.device)
        res_mat = res_mat + node_ft

        # generate adj
        h_hat = self.adj_trans_layer(node_ft)
        w = torch.norm(h_hat, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        adj = torch.mm(h_hat, h_hat.t()) / w_mat

        for idx in range(self.gcn_layer_num):
            node_ft = self.gcn_list[idx](node_ft, adj)
            node_ft = self.activation(node_ft)
            node_ft = F.dropout(node_ft, p=self.dropout)
            res_mat = res_mat + node_ft

        out = self.linear_out(res_mat)
        return out, adj


from dataset.load_dataset import GlobalFlu
import torch
import torch.nn.functional as F
import numpy as np
import random
from tools.evaluate_utils import evaluate_regression


class VirtualGCNTrainer(object):
    def __init__(self, wind_size=52, pred_step=1, data_type='us', split_param=[0.6, 0.2, 0.2],
                 seed=3, gcn_layer_num=2, dropout=0.4):
        self.setup_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('VGCN')
        self.dataset = GlobalFlu(wind_size=wind_size, pred_step=pred_step, data_type=data_type, split_param=split_param)
        self.dataset.to_tensor()
        self.dataset.to_device(self.device)

        self.epochs = 200
        self.count = 0
        self.loss_type = 'mse'
        self.dropout = dropout
        self.gcn_layer_num = gcn_layer_num
        self.pred_nums = None
        self.min_loss = 1e10
        self.batch_size = 30
        self.build_model()

    def build_model(self):
        self.model = Model(
            node_num=self.dataset.node_size,
            in_channels=self.dataset.node_feature_size,
            out_channels=self.dataset.label_size,
            gcn_layer_num=self.gcn_layer_num,
            dropout=self.dropout,
            bias=True
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.best_res = 0
        self.adj = None

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

            out, adj = self.model(self.dataset.ft_mat)
            out = out[batch_idx]

            reg_loss = 0
            if self.loss_type == 'mse':
                reg_loss = F.mse_loss(out, batch_node_label, reduction='mean')
            if self.loss_type == 'mape':
                reg_loss = self.mape_loss(out, batch_node_label)
            if self.loss_type == 'mae':
                reg_loss = self.mae_loss(out, batch_node_label)

            l1_reg = 0
            for name, param in self.model.named_parameters():
                if 'bias' not in name:
                    l1_reg += torch.norm(param, p=1)
            l1_loss = 5e-5 * l1_reg
            loss = reg_loss + l1_loss

            loss.backward()
            self.optimizer.step()
        # print(adj)

    def test(self):
        self.model.eval()
        pred, adj = self.model(self.dataset.ft_mat)
        label = self.dataset.label_mat.to('cpu').detach().numpy()
        pred = pred.to('cpu').detach().numpy()

        train_mse, train_mae, train_mape = evaluate_regression(
            pred[self.dataset.train_index], label[self.dataset.train_index])

        valid_mse, valid_mae, valid_mape = evaluate_regression(
            pred[self.dataset.valid_index], label[self.dataset.valid_index])

        test_mse, test_mae, test_mape = evaluate_regression(
            pred[self.dataset.test_index], label[self.dataset.test_index])

        return train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
               train_mape, valid_mape, test_mape, pred, adj

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
        for epoch in range(1, self.epochs):
            self.train()
            train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
            train_mape, valid_mape, test_mape, pred, adj = self.test()

            if valid_mse < self.min_loss:
                self.min_loss = valid_mse
                self.pred_nums = pred
                self.best_res = [train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
                                 train_mape, valid_mape, test_mape]
                self.adj = adj.to('cpu').detach().numpy()
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
    seed = 3
    for pred_step in [1, 3, 6]:
        for data_type in ['us']:
            for wind_size in [6, 9, 12]:
                res = VirtualGCNTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type, seed=seed, dropout=0.1,
                                  gcn_layer_num=2).start(display=False)
