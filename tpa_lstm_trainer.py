import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Parameter


class Model(nn.Module):
    def __init__(self,
                 node_num,
                 seq_len,
                 out_channels,
                 mid_channels=256,
                 ):
        super(Model, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        # 单层 LSTM
        self.lstm = nn.LSTM(1, self.mid_channels, 1)
        self.linear_mid = nn.Linear(self.mid_channels, self.mid_channels)
        self.linear_mid2 = nn.Linear(self.mid_channels, self.mid_channels)
        self.linear_out = nn.Linear(self.mid_channels, self.out_channels)

        # attention
        self.att_linear1 = nn.Linear(self.mid_channels, 1)
        self.att_linear2 = nn.Linear(self.mid_channels, 1)

        self.mlp_activation = nn.ELU()

    def forward(self, node_ft):
        node_num = node_ft.size()[0]
        seq_len = node_ft.size()[1]
        input_mat = node_ft.transpose(0, 1)
        input_mat = input_mat.view(seq_len, node_num, 1)

        batch_size = node_num
        h0 = torch.randn(1, batch_size, self.mid_channels).to(node_ft.device)
        c0 = torch.randn(1, batch_size, self.mid_channels).to(node_ft.device)
        output, (hn, cn) = self.lstm(input_mat, (h0, c0))
        output = output.transpose(0, 1) # batch, wind_size, hidden_size
        # attention
        f1 = self.att_linear1(output)
        f2 = self.att_linear2(output)
        logits = f1 + torch.transpose(f2, 2, 1)
        coefs = F.softmax(F.leaky_relu(logits, negative_slope=0.2), dim=-1)
        H = torch.bmm(coefs, output)[:,-1,:]  # 只要最后一个时刻
        H = self.mlp_activation(H)
        pred = self.linear_out(H)

        return pred


from dataset.load_dataset import GlobalFlu

import torch
import torch.nn.functional as F
import numpy as np
import random
from tools.evaluate_utils import evaluate_regression


class LSTMTrainer(object):
    def __init__(self, wind_size=52, pred_step=1, layer_num=2, data_type='us', split_param=[0.6, 0.2, 0.2], seed=3):
        self.setup_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('TPA LSTM')
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
        self.model = Model(
            node_num=self.dataset.node_size,
            seq_len=self.dataset.node_feature_size,
            out_channels=self.dataset.label_size
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

            out = self.model(self.dataset.ft_mat[batch_idx])

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
  for seed in [3]:
  # for seed in range(10):
    print('seed = ', seed)
    mse_res_list = []
    mape_res_list = []
    for pred_step in [1, 3, 6]:
        for data_type in ['us']:
            for wind_size in [6, 9, 12]:
                res = LSTMTrainer(wind_size=wind_size, pred_step=pred_step, data_type=data_type, seed=seed, layer_num=2).start(display=False)
                mse_res_list.append(res[0])  # mse
                mape_res_list.append(res[2])  # mape

