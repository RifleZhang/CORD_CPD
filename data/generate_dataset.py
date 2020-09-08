from synthetic_cp import SpringSim
import time
import numpy as np
import argparse
import os, sys, time
import os.path as osp
from tqdm import tqdm
import copy

def load_file(path, suffix, mode='train'):
    # loc: [batch, time_steps, dim, num_obj]
    loc = np.load(osp.join(path, "cp_loc_{}_{}.npy".format(mode, suffix)))
    vel = np.load(osp.join(path, "cp_vel_{}_{}.npy".format(mode, suffix)))
    # feature: [batch, time_steps, dim*2, num_obj]
    feature = np.concatenate([loc, vel], axis=2)
    res = [feature]
    files = ['change_point', "edges1", "edges2"]
    for f in files:
        fp = osp.join(path, "cp_{}_{}_{}.npy".format(f, mode, suffix))
        res.append(np.load(fp))
    return res

def save_file(data, path, suffix, mode='train'):
    if not osp.exists(path):
        os.makedirs(path)
    files = ['feature', 'change_point', "edges1", "edges2"]
    for i, f in enumerate(files):
        fp = osp.join(path, "cp_{}_{}_{}.npy".format(f, mode, suffix))
        np.save(fp, data[i])
    
def merge_train(paths, save_name = "cp_small_whole"):
    data = []
    suffix = "variable_5"
    for i in range(len(paths)):
        tmp = load_file(paths[i], suffix, 'train')
        if len(data) == 0:
            data = tmp
        else:
            for j in range(len(data)):
                data[j] = np.concatenate([data[j], tmp[j]])
    save_file(data, save_name, suffix, 'train')
    return data

def merge_valid_test(paths, mode='valid', save_name = 'cp_small_whole'):
    data = []
    suffix = "variable_5"
    for i in range(len(paths)):
        tmp = load_file(paths[i], suffix, mode)
        sample_idx = np.random.choice(len(tmp[0]), 50, replace=False)
        for i in range(len(tmp)):
            tmp[i] = tmp[i][sample_idx]
        if len(data) == 0:
            data = tmp
        else:
            for j in range(len(data)):
                data[j] = np.concatenate([data[j], tmp[j]])
    save_file(data, save_name, suffix, mode)
    return data

paths = ["cp_loc", "cp_vel", "cp_edge"]
name = "cp_change"
data = merge_train(paths, name)
# logic for valid and test are the same
data_val = merge_valid_test(paths, 'valid', save_name=name)
data_val = merge_valid_test(paths, 'test', save_name=name)

















