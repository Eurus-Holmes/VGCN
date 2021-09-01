import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import Parameter

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        res = self.network(x)
        # print('res.size() = ', res.size())
        return res


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__()
        # self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], output_size)
        # self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        # emb = self.drop(self.encoder(x))
        x = torch.unsqueeze(x, dim=-1)
        # print(x.size())
        x = x.transpose(1, 2)

        y = self.tcn(x)
        # print(y.size())
        y = y[:,:,-1]
        # print(y.size())
        o = self.decoder(y)
        # print('o.size() = ', o.size())
        return o.contiguous()


from dataset.load_dataset import GlobalFlu

import torch
import torch.nn.functional as F
import numpy as np
import random
from tools.evaluate_utils import evaluate_regression


class TCNTrainer(object):
    def __init__(self, wind_size=52, pred_step=1, data_type='us', split_param=[0.6, 0.2, 0.2], seed=3):
        self.setup_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('TCN')
        self.dataset = GlobalFlu(wind_size=wind_size, pred_step=pred_step, data_type=data_type, split_param=split_param)
        self.dataset.to_tensor()
        self.dataset.to_device(self.device)
        self.pred_step = pred_step

        self.epochs = 200
        self.count = 0
        self.loss_type = 'mse'

        self.alpha = 0.001
        self.pred_nums = None
        self.min_loss = 1e10
        self.batch_size = self.dataset.train_index.shape[0]
        # self.batch_size = 30
        # print(self.dataset.node_ft_mat.size())
        self.build_model()

    def build_model(self):
        self.model = TCN(1, self.pred_step, [64, 64], kernel_size=2, dropout=0.2, emb_dropout=0.2
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
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
        # print(train_size)
        shuffle_idx = torch.randperm(train_size)
        for idx in range(train_size // self.batch_size):
            self.optimizer.zero_grad()
            right_bound = min((idx + 1) * self.batch_size, train_size + 1)
            shuffle_batch_idx = shuffle_idx[idx * self.batch_size: right_bound]
            batch_idx = self.dataset.train_index[shuffle_batch_idx]
            batch_node_label = self.dataset.label_mat[batch_idx]
            out = self.model(self.dataset.ft_mat)
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

        pred = self.model(self.dataset.ft_mat)
        label = self.dataset.label_mat.to('cpu').detach().numpy()
        pred = pred.to('cpu').detach().numpy()

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
            loss = valid_mse

            if loss < self.min_loss:
                self.min_loss = loss
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

    mse_res_list = []
    mape_res_list = []
    for pred_step in [1, 3, 6]:
        for data_type in ['us', 'ch_north', 'ch_south']:
            for wind_size in [6, 9, 12]:
                res = TCNTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type, seed=3,
                                  ).start(display=True)
                mse_res_list.append(res[0])  # mse
                mape_res_list.append(res[2])  # mape

