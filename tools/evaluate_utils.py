import numpy as np
import torch

def evaluate_regression(pred, label):
    '''
    :param pred: numpy
    :param label: numpy
    :return:
    '''
    if type(pred) == torch.Tensor:
        pred = pred.detach().to('cpu').numpy()
    if type(label) == torch.Tensor:
        label = label.detach().to('cpu').numpy()

    mse = np.mean(np.square(pred - label))

    mae = np.mean(np.abs(pred - label))

    mape = np.mean(np.abs((pred - label) / label))
    return mse, mae, mape


