import numpy as np
from torch.utils import data
import torch
from sklearn.metrics import roc_auc_score

def computeAUC(predictions, labels):
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    elif labels.shape[1] == 1:
        labels = np.eye(2, dtype='uint8')[labels].squeeze()
    else:
        pass
    return roc_auc_score(labels, predictions[:, 1])

def exponentialDecay(N):
    tau = 1
    tmax = 7
    t = np.linspace(0, tmax, N)
    y = torch.tensor(np.exp(-t/tau), dtype=torch.float)
    return y

def createNet(n_inputs, n_outputs, n_layers=0, n_units=100, nonlinear=torch.nn.Tanh):
    if n_layers == 0:
        return torch.nn.Linear(n_inputs, n_outputs)
    layers = [torch.nn.Linear(n_inputs, n_units)]
    for i in range(n_layers-1):
        layers.append(nonlinear())
        layers.append(torch.nn.Linear(n_units, n_units))
        layers.append(torch.nn.Dropout(p=0.5))

    layers.append(nonlinear())
    layers.append(torch.nn.Linear(n_units, n_outputs))
    return torch.nn.Sequential(*layers)
