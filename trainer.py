import math
import sys, os
import os.path as osp
import time
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.exp_utils import create_exp_dir
from utils.utils import *
from utils.cp_data_utils import *
from utils.cp_function_utils import *
from models import cord_cpd

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        self.args.cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.args.cuda else torch.device("cpu")
        self.args.factor = not args.no_factor
        self.exp_dir = args.exp_dir

        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.logging = create_exp_dir(args.exp_dir)

        meta_file_name = osp.join(args.exp_dir, "meta.txt")
        meta_file = open(meta_file_name, "w")
        meta_file.write(str(args))
        meta_file.close()

    def load_data(self):
        args = self.args
        if self.data_type == "sim":
            self.train_loader, self.valid_loader, self.test_loader = load_cp_data(
            args.data_path, args.batch_size, args.suffix, args.data_norm)
        else:
            self.train_loader, self.valid_loader, self.test_loader = load_cp_real_data(
                args.data_path, args.batch_size)
        off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)

    def set_model(self):
        self.model = cord_cpd.Model(self.args)
        self.model.to_device(self.device)
        if self.args.load:
            self.logging("loading model from {}".format(self.args.exp_dir))
            self.model.load(self.exp_dir)

    def train(self):
        # Train model
        st = time.time()
        best_val_loss = np.inf
        best_acc_val = 0
        best_epoch = 0

        for epoch in range(self.args.epochs):
            t = time.time()
            mse_train, delta_train, acc_train = self.train_one_epoch()
            log_str_train = "Epoch: {:4d}, mse_train: {:.4f}, delta_train: {:.4f}, " \
                      "acc_train: {:.4f}, epoch time: {:.2f}s".format(
                epoch, mse_train, delta_train,
                acc_train, time.time() - t
            )

            log_str_eval = ""
            if (epoch+1) % self.args.eval_epoch == 0:
                mse_val, delta_val, acc_val, avg_roc, avg_dist, avg_tri = self.evaluate()

                log_str_eval = "|| mse_val: {:.4f}, delta_val: {:.4f}, acc_val: {:.4f}, " \
                          "roc: {:.4f}, dist: {:.4f}, tri: {:.4f}, total time: {:.2f}s ||".format(
                    mse_val, delta_val, acc_val,
                    avg_roc, avg_dist, avg_tri, time.time()-st
                )
                if mse_val < best_val_loss:
                    best_val_loss = mse_val
                    best_epoch = epoch
                    self.model.save(self.exp_dir)
                    self.logging("save best model at epoch : {}".format(best_epoch))
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.model.save(self.exp_dir, "acc_model.t7")
                    self.logging("save acc model at epoch : {}".format(epoch))
            self.logging(log_str_train + log_str_eval)

        self.logging("Optimization Finished!")
        self.logging("Best Epoch: {:04d}".format(best_epoch))

    def train_one_epoch(self):
        acc_train = []
        mse_train = []
        delta_train = []

        self.model.set_train()
        for batch_idx, (data, relations, cpd) in enumerate(self.train_loader):

            data, relations = data.to(self.device), relations.to(self.device)
            # data [batch_size, num_atoms, num_timesteps, num_dims]

            data = data[:, :, :self.args.timesteps, :]

            self.model.optimizer.zero_grad()

            logits = self.model.encode(data, self.rel_rec, self.rel_send)
            # loss_delta = 10000 * ((logits[:, :-1] - logits[:, 1:]) ** 2).mean()
            # logits [batch, timestep, edge, relation]
            sub_logits = logits[:, 5:-5]
            loss_delta = 100 * ((sub_logits[:, :-1] - sub_logits[:, 1:]) ** 2).mean()
            edges = F.gumbel_softmax(logits, tau=self.args.temp, hard=self.args.hard)
            # prob = F.softmax(logits, -1)

            output = self.model.decode(data, edges, self.rel_rec, self.rel_send)

            target = data[:, :, 1:, :]

            target = target[:, :, self.args.begin_steps:, :]
            output = output[:, :, self.args.begin_steps:, :]

            loss_mse = F.mse_loss(output, target) / (2 * self.args.var) * 400

            loss = loss_mse + loss_delta
            if self.data_type == "sim":
                acc = edge_accuracy(logits, relations, begin_steps=5, end_steps=-5)
                acc_train.append(acc)
            else:
                acc_train.append(np.nan)

            loss.backward()
            self.model.optimizer.step()

            mse_train.append(loss_mse.item())
            delta_train.append(loss_delta.item())
        self.model.scheduler.step()

        return np.mean(mse_train), np.mean(delta_train), np.mean(acc_train)

    @torch.no_grad()
    def evaluate(self):
        acc_val = []
        mse_val = []
        delta_val = []

        self.model.set_eval()
        probs = []
        cpds = []
        recons = []
        origs = []
        for batch_idx, (data, relations, cpd) in enumerate(self.valid_loader):
            data, relations = data.to(self.device), relations.to(self.device)
            data = data[:, :, :self.args.timesteps, :]

            logits = self.model.encode(data, self.rel_rec, self.rel_send)
            sub_logits = logits[:, 5:-5]
            loss_delta = 100 * ((sub_logits[:, :-1] - sub_logits[:, 1:]) ** 2).mean()
            edges = F.gumbel_softmax(logits, tau=self.args.temp, hard=True)
            prob = F.softmax(logits, -1)
            probs.append(prob)
            cpds.extend(cpd)

            output = self.model.decode(data, edges, self.rel_rec, self.rel_send)
            target = data[:, :, 1:, :]

            target = target[:, :, self.args.begin_steps:, :]
            output = output[:, :, self.args.begin_steps:, :]

            loss_mse = F.mse_loss(output, target) / (2 * self.args.var) * 400

            if self.data_type == 'sim':
                acc = edge_accuracy(logits, relations, begin_steps=5, end_steps=-5)
                acc_val.append(acc)
            else:
                origs.append(data.transpose(1, 2).contiguous().detach().cpu().numpy())

                # validation output uses teacher forcing
                recon = self.model.decoder.forward_reconstruct(data, edges, self.rel_rec, self.rel_send)
                recons.append(recon.detach().cpu().numpy())
                acc_val.append(np.nan)

            mse_val.append(loss_mse.item())
            delta_val.append(loss_delta.item())

        probs = torch.cat(probs).detach().cpu().numpy()
        cpds = np.array(cpds)
        avg_roc, avg_dist, avg_tri = cpd_metrics(probs, cpds)

        if self.report_combine:
            recons = np.concatenate(recons)
            origs = np.concatenate(origs)
            type1_score = mse_anomaly(recons, origs, step=5)
            type2_score = cal_cp_from_output(probs)
            combined = anomaly_combined_score(type1_score, type2_score)
            self.logging("-"*30)
            self.logging("relation score: {}".format(cpd_metrics(type2_score, cpds, anomaly_input=True)))
            self.logging("mse score: {}".format(cpd_metrics(type1_score, cpds, anomaly_input=True)))
            self.logging("combined score: {}".format(cpd_metrics(combined, cpds, anomaly_input=True)))
            self.logging("-" * 30)

        self.model.set_train()
        return np.mean(mse_val), np.mean(delta_val), np.mean(acc_val), avg_roc, avg_dist, avg_tri
