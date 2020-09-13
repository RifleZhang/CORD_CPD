import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import os, sys, os.path as osp

eps = 1e-8

# normalize data
def min_max_transform(train_fea, valid_fea, test_fea):
    def _transform(fea, fmax, fmin):
        for i in range(4):
            fea[:, :, i] = (fea[:, :, i] - fmin[i]) * 2 / (fmax[i] - fmin[i]) - 1
        return fea

    # hard code feature dim 2 with 4 features
    fmax, fmin = np.zeros(4), np.zeros(4)
    for i in range(4):
        fmax[i], fmin[i] = train_fea[:, :, i].max(), train_fea[:, :, i].min()
    print("min max norm: ", fmax, fmin)
    return [_transform(f, fmax, fmin) for f in [train_fea, valid_fea, test_fea]]

def mean_std_transform(train_fea, valid_fea, test_fea):
    def _transform(fea, u, s):
        for i in range(4):
            fea[:, :, i] = (fea[:, :, i] - u[i]) / s[i]
        return fea

    # hard code feature dim 2 with 4 features
    u, s = np.zeros(4), np.zeros(4)
    for i in range(4):
        u[i], s[i] = train_fea[:, :, i].mean(), train_fea[:, :, i].std()
    print("mean std norm: ", u, s)
    return [_transform(f, u, s) for f in [train_fea, valid_fea, test_fea]]

def make_edges(data, total_length, num_var):
    # cp [batch]
    # edge1,2 [batch, num_var, num_var]
    cp, edge1, edge2 = data
    # edge -> [batch_size, num_edge]
    edge1 = edge1.reshape([-1, num_var ** 2]).astype(int)
    edge2 = edge2.reshape([-1, num_var ** 2]).astype(int)

    # cp [batch_size]
    res = []
    # concat [time_step, edge_num]
    for i in range(len(edge1)):
        try:
            concat = np.concatenate(
                [edge1[i:i + 1].repeat(cp[i], axis=0), edge2[i:i + 1].repeat(total_length - cp[i], axis=0)], 0)
        except:
            # no perturbation of edges
            concat = edge1[i:i + 1].repeat(total_length, axis=0)
        res.append(concat)
    res = np.array(res)
    return res

def load_file(path, suffix, mode='train'):
    files = ['change_point', "edges1", "edges2"]
    try:
        fp = osp.join(path, "cp_feature_{}_{}.npy".format(mode, suffix))
        feature = np.load(fp)
    except:
        # batch, time_steps, dim, num_var
        loc = np.load(osp.join(path, "cp_loc_{}_{}.npy".format(mode, suffix)))
        vel = np.load(osp.join(path, "cp_vel_{}_{}.npy".format(mode, suffix)))
        feature = np.concatenate([loc, vel], axis=2)

    res = [feature]
    for f in files:
        fp = osp.join(path, "cp_{}_{}_{}.npy".format(f, mode, suffix))
        res.append(np.load(fp))
    return res


def load_cp_data(path, batch_size=1, suffix='variable_5', data_norm="mean_std"):
    modes = ["train", "valid", "test"]
    # data [feature, change_point, edge1, edge2]
    train, valid, test = [load_file(path, suffix, mode) for mode in modes]

    # normalize feature
    # train [feature, change_point, edges, edges2]
    # train_fea [batch, time_step, dim, num_var]
    if data_norm == 'min_max':
        train_fea, valid_fea, test_fea = min_max_transform(train[0], valid[0], test[0])
    else:
        train_fea, valid_fea, test_fea = mean_std_transform(train[0], valid[0], test[0])
    train_fea, valid_fea, test_fea = [o.transpose(0, 3, 1, 2) for o in [train_fea, valid_fea, test_fea]]
    # train_fea [batch, num_var, time_steps, dim]

    # process edge data to be edge connection per time step
    batch, num_var, time_steps, dim = train_fea.shape
    # train_edge [batch, time_step, num_edge]
    train_edge = make_edges(train[1:], time_steps, num_var)
    valid_edge = make_edges(valid[1:], time_steps, num_var)
    test_edge = make_edges(test[1:], time_steps, num_var)

    train_fea, valid_fea, test_fea = [torch.FloatTensor(o) for o in [train_fea, valid_fea, test_fea]]
    train_edge, valid_edge, test_edge = [torch.LongTensor(o) for o in [train_edge, valid_edge, test_edge]]
    train_cp, valid_cp, test_cp = [torch.LongTensor(o) for o in [train[1], valid[1], test[1]]]

    # Exclude self edges by selecting off diag index
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_var, num_var)) - np.eye(num_var)),
        [num_var, num_var])
    # train_edge [batch, time_step, num_non-self_edge]
    train_edge = train_edge[:, :, off_diag_idx]
    valid_edge = valid_edge[:, :, off_diag_idx]
    test_edge = test_edge[:, :, off_diag_idx]

    train_data = TensorDataset(train_fea, train_edge, train_cp)
    valid_data = TensorDataset(valid_fea, valid_edge, valid_cp)
    test_data = TensorDataset(test_fea, test_edge, test_cp)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def norm_time_series(ts):
    # norm single dim
    fmean = np.mean(ts, 0)
    fstd = np.std(ts, 0)
    norm_ts = (ts - fmean) / (fstd + eps)
    return norm_ts, fmean, fstd

def transform(ts, fmean, fstd):
    return (ts - fmean) / fstd

def load_real_file(path, mode='train'):
    files = ["Features", "ChangePoints", "Targets"]
    res = []
    for f in files:
        fp = osp.join(path, "{}{}.npy".format(mode, f))
        res.append(np.load(fp))
    return res

def load_cp_real_data(path, batch_size=1, data_norm="mean_std"):
    modes = ["train", "valid", "test"]
    # data [feature, change_point, edge1, edge2]
    train, valid, test = [load_real_file(path, mode) for mode in modes]

    # train_fea [batch, time_step, dim, mum_atom]
    if data_norm == 'min_max':
        train_fea, valid_fea, test_fea = min_max_transform(train[0], valid[0], test[0])
    else:
        train_fea, valid_fea, test_fea = mean_std_transform(train[0], valid[0], test[0])
    train_fea, valid_fea, test_fea = [o.transpose(0, 3, 1, 2) for o in [train_fea, valid_fea, test_fea]]
    # train_fea [batch, num_atom, time_steps, dim]


    # edge
    train_fea, valid_fea, test_fea = [torch.FloatTensor(o) for o in [train_fea, valid_fea, test_fea]]
    train_edge, valid_edge, test_edge = [torch.zeros(o.shape) for o in [train_fea, valid_fea, test_fea]]
    train_cp, valid_cp, test_cp = [torch.LongTensor(o) for o in [train[1], valid[1], test[1]]]

    train_data = TensorDataset(train_fea, train_edge, train_cp)
    valid_data = TensorDataset(valid_fea, valid_edge, valid_cp)
    test_data = TensorDataset(test_fea, test_edge, test_cp)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader











