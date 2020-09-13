import time
import argparse
import pickle
import os
import datetime

from trainer import Trainer
from tester import Tester

def parse_args(cmd=None):
    parser = argparse.ArgumentParser()

    # experiment setting
    parser.add_argument('--mode', type=str, choices=['train', 'test'])
    parser.add_argument('--exp_dir', type=str, default='exp/exp')
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--load', action='store_true', help='for fine tunning. Leave empty to train from scratch')

    # data setting
    parser.add_argument('--data_norm', type=str, default="mean_std")
    parser.add_argument('--data_path', type=str, default="data/cp_change")
    parser.add_argument('--data_type', type=str, default="sim")

    # training setting
    parser.add_argument('--epochs', type=int, default=50000,
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--eval_epoch', type=int, default=5)

    # model
    parser.add_argument('--spatial-encoding-layer', type=str, default="gnn", choices=["gnn", "transformer"],
                        help='spatial encoder')
    parser.add_argument('--temporal-encoding-layer', type=str, default="transformer", choices=["rnn", "transformer"],
                        help='temporal encoder')
    parser.add_argument('--decoder', type=str, default='mlp')
    parser.add_argument('--dims', type=int, default=4,
                        help='The number of input dimensions (position + velocity).')
    parser.add_argument('--encoder-hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--decoder-hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--edge-types', type=int, default=2,
                        help='The number of edge types to infer.')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')
    parser.add_argument('--num-atoms', type=int, default=5,
                        help='Number of atoms in simulation.')
    parser.add_argument('--no-factor', action='store_true', default=False,
                        help='Disables factor graph model.')
    parser.add_argument('--suffix', type=str, default='variable_5')
    parser.add_argument('--encoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--decoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--timesteps', type=int, default=100,
                        help='The number of time steps per sample.')
    parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                        help='Num steps to predict before re-using teacher forcing.')
    parser.add_argument('--begin-steps', type=int, default=0,
                        help='Num steps begin to predict')
    parser.add_argument('--lr-decay', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--skip-first', action='store_true', default=True,
                        help='Skip first edge type in decoder, i.e. it represents no-edge.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='Uses discrete samples in training forward pass.')
    parser.add_argument('--dynamic-graph', action='store_true', default=False,
                        help='Whether test with dynamically re-computed graph.')


    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd.split())

    print(args)

    # if args.dynamic_graph:
    #     print("Testing with dynamically re-computed graph.")
    return args

def main():
    args = parse_args()

    print("exp name: {}".format(args.exp_dir))
    if args.mode == 'train':
        train = Trainer(args)
        train.data_type = args.data_type
        train.report_combine = False # set to True if using real data
        train.logging("process data")
        train.load_data()
        train.logging("set model and train")
        train.set_model()
        train.train()
    elif args.mode == 'test':
        test = Tester(args)
        test.solve()
    else:
        raise RuntimeError(f'invalid mode')

if __name__ == '__main__':
    main()