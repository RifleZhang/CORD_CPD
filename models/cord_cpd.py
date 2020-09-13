import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import os, sys
import os.path as osp
import numpy as np

from models.encoder_modules import *
from models.decoder_modules import *

class Model(object):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.dec_type = args.decoder
        self.timesteps = args.timesteps
        self.prediction_steps = args.prediction_steps

        self.encoder = CorrelationEncoder(args.dims, args.encoder_hidden,
                                 args.edge_types, args.spatial_encoding_layer, args.temporal_encoding_layer,
                                 args.encoder_dropout, args.factor)

        if args.decoder == 'mlp':
            self.decoder = MLPDecoder(n_in_node=args.dims,
                                 edge_types=args.edge_types,
                                 msg_hid=args.decoder_hidden,
                                 msg_out=args.decoder_hidden,
                                 n_hid=args.decoder_hidden,
                                 do_prob=args.decoder_dropout,
                                 skip_first=args.skip_first)
        elif args.decoder == 'rnn':
            self.decoder = RNNDecoder(n_in_node=args.dims,
                                 edge_types=args.edge_types,
                                 n_hid=args.decoder_hidden,
                                 do_prob=args.decoder_dropout,
                                 skip_first=args.skip_first)

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                               lr=args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay,
                                        gamma=args.gamma)

    def encode(self, data, rel_rec, rel_send):
        return self.encoder(data, rel_rec, rel_send)

    def decode(self, data, edges, rel_rec, rel_send, burn_in=True, burn_in_steps=None):
        if self.dec_type == 'rnn':
            if not burn_in:
                burn_in_steps = 0
            elif burn_in_steps is None:
                burn_in_steps = self.timesteps - self.prediction_steps
            output = self.decoder(data, edges, rel_rec, rel_send, self.timesteps,
                         burn_in=burn_in,
                         burn_in_steps=burn_in_steps)
        else:
            output = self.decoder(data, edges, rel_rec, rel_send, self.prediction_steps)
        return output

    def set_train(self):
        self.encoder.train()
        self.decoder.train()

    def set_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def to_device(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def save(self, exp_path, name="model.t7"):
        model_path = osp.join(exp_path, name)
        torch.save({
            'enc_state_dict': self.encoder.state_dict(),
            'dec_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, model_path)

    def load(self, model_path):
        model_path = osp.join(model_path, 'model.t7')
        checkpoint = torch.load(model_path)
        self.encoder.load_state_dict(checkpoint['enc_state_dict'])
        self.decoder.load_state_dict(checkpoint['dec_state_dict'])
        if self.args.mode=='train':
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
