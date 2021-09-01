import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import Parameter

class DotAtt(nn.Module):
    def __init__(self,
                 scale=64):
        super(DotAtt, self).__init__()
        self.scale = scale

    def forward(self, q_mat, k_mat, v_mat):
        # ft_mat  batch * win_size * ft_size
        weight_mat = torch.bmm(q_mat, k_mat.transpose(1, 2)) * self.scale
        weight_mat = torch.softmax(weight_mat, dim=-2)

        weight_mat = F.dropout(weight_mat, p=0.1, training=True)
        ft_mat = torch.bmm(weight_mat, v_mat)
        return ft_mat

class TransformerLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 wind_size=3,
                 heads=4,
                 dropout=0.3,
                 scale=0.1):
        super(TransformerLayer, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.scale = scale
        self.heads = heads
        self.dropout = dropout
        self.wind_size = wind_size

        self.q_linear = nn.Linear(in_channels, mid_channels * self.heads)
        self.k_linear = nn.Linear(in_channels, mid_channels * self.heads)
        self.v_linear = nn.Linear(in_channels, mid_channels * self.heads)

        # 传到下一层
        self.final_linear = nn.Linear(mid_channels * self.heads, mid_channels)

        # 加入 res_net
        self.out_linear = nn.Linear(mid_channels * self.wind_size, out_channels)

        self.att_layer = DotAtt(scale=self.scale)
        self.norm_layer = nn.LayerNorm(mid_channels)

    def forward(self, ft_mat):
        # batch * win_size * fr_size(1)
        batch_size = ft_mat.size()[0]

        q_mat = self.q_linear(ft_mat)
        k_mat = self.k_linear(ft_mat)
        v_mat = self.v_linear(ft_mat)  # batch * win_size * [ft_size * heads]

        q_mat = q_mat.view(batch_size, -1, self.mid_channels, self.heads).transpose(1, -1).transpose(-2, -1)
        k_mat = k_mat.view(batch_size, -1, self.mid_channels, self.heads).transpose(1, -1).transpose(-2, -1)
        v_mat = v_mat.view(batch_size, -1, self.mid_channels, self.heads).transpose(1, -1).transpose(-2, -1)
        # print(q_mat.size())
        q_mat = q_mat.reshape(batch_size * self.heads, -1, self.mid_channels)
        k_mat = k_mat.reshape(batch_size * self.heads, -1, self.mid_channels)
        v_mat = v_mat.reshape(batch_size * self.heads, -1, self.mid_channels)

        v_mat = self.att_layer(q_mat, k_mat, v_mat)

        ft_mat = v_mat.view(batch_size, self.heads, -1, self.mid_channels).transpose(1, 2)
        ft_mat = ft_mat.reshape(batch_size, -1, self.heads * self.mid_channels)
        ft_mat = self.final_linear(ft_mat)
        tmp = F.elu(ft_mat.view(batch_size, -1))
        out = self.out_linear(tmp)

        return ft_mat, out


class Transformer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 heads=4,
                 dropout=0.1,
                 layer_num=3,
                 wind_size=3,
                 scale=0.1,
                 ):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.heads = heads
        self.dropout = dropout
        self.layer_num = layer_num
        self.wind_size = wind_size

        self.transformer_layers = nn.ModuleList()
        for idx in range(self.layer_num):
            in_size = mid_channels
            out_size = out_channels
            if idx == 0:
                in_size = 1
            if idx == self.layer_num - 1:
                out_size = out_channels

            tmp_layer = TransformerLayer(in_channels=in_size,
                                         out_channels=out_size,
                                         mid_channels=self.mid_channels,
                                         wind_size=self.wind_size,
                                         dropout=self.dropout, heads=self.heads)
            self.transformer_layers.append(tmp_layer)

    def forward(self, ft_mat):
        # ft_mat -- batch_size * win_size * 1
        batch_size = ft_mat.size()[0]
        out = torch.zeros(batch_size, self.out_channels).to(ft_mat.device)

        for idx in range(self.layer_num):
            ft_mat, layer_out = self.transformer_layers[idx](ft_mat)
            ft_mat = F.elu(ft_mat)
            out = out + layer_out
        return out

from tools.evaluate_utils import evaluate_regression
from dataset.load_dataset import  GlobalFlu
import torch
import torch.nn.functional as F
import numpy as np
import random


class TransformerTrainer(object):
    def __init__(self, wind_size=52, pred_step=1, data_type='us', split_param=[0.6, 0.2, 0.2], seed=3, layer_num=2):
        self.setup_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.win_size = wind_size

        self.dataset = GlobalFlu(wind_size=wind_size, pred_step=pred_step, data_type=data_type, split_param=split_param)
        self.dataset.to_tensor()
        self.dataset.to_device(self.device)


        self.epochs = 200
        self.count = 0
        self.loss_type = 'mse'

        self.alpha = 0.001
        self.layer_num = layer_num
        self.pred_nums = None
        self.min_loss = 1e10
        self.batch_size = 30
        self.build_model()

    def build_model(self):
        self.model = Transformer(
            in_channels=self.dataset.node_feature_size,
            out_channels=self.dataset.label_size,
            mid_channels=256,
            layer_num=self.layer_num,
            wind_size=self.win_size,
            heads=1,
            dropout=0.0,
            scale=0.3
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
        for idx in range( train_size // self.batch_size):
            self.optimizer.zero_grad()
            right_bound = min((idx + 1) * self.batch_size, train_size + 1)
            shuffle_batch_idx = shuffle_idx[idx * self.batch_size: right_bound]
            batch_idx = self.dataset.train_index[shuffle_batch_idx]
            batch_node_label = self.dataset.label_mat[batch_idx]
            ft_mat = self.dataset.ft_mat[batch_idx]

            ft_mat = ft_mat.view(self.batch_size, -1, 1)
            out = self.model(ft_mat)
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

        ft_mat = self.dataset.ft_mat
        ft_mat = ft_mat.view(self.dataset.node_size, -1, 1)
        pred = self.model(ft_mat).to('cpu').detach().numpy()
        label = self.dataset.label_mat.to('cpu').detach().numpy()

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

    def start(self, topn=20, display=True):
        self.test_acc_list = []
        for epoch in range(1, self.epochs):
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
        self.print_best_res()
        train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae, \
        train_mape, valid_mape, test_mape, epoch = self.best_res
        return test_mse, test_mae, test_mape

if __name__ == '__main__':
    # res = TransformerTrainer(wind_size=6, pred_step=3, data_type='us', seed=3, layer_num=2).start(display=True)
    res_list = []
    for pred_step in [1, 3, 6]:
        for data_type in ['us']:
            for wind_size in [6, 9, 12]:
                res = TransformerTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type, seed=3).start(display=False)
                res_list.append(res[0]) # mse



