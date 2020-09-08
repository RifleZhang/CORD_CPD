from synthetic_cp import SpringSim
import time
import numpy as np
import argparse
import os, os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--out', type=str, default="cp_data")
parser.add_argument('--change-type', type=str, default='edge')


args = parser.parse_args()


sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
suffix = 'variable_' + str(args.n_balls)
np.random.seed(args.seed)

print(f"simulating {args.n_balls} objects".format(args.n_balls))

def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges1_all = list()
    edges2_all = list()
    change_point_all = list()

    print("Change type:", args.change_type)

    for i in range(num_sims):
        t = time.time()
        loc, vel, change_point, edges1, edges2 = sim.sample_trajectory(T=length,
            sample_freq=sample_freq, change_type=args.change_type)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        change_point_all.append(change_point)
        edges1_all.append(edges1)
        edges2_all.append(edges2)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    change_point_all = np.stack(change_point_all)
    edges1_all = np.stack(edges1_all)
    edges2_all = np.stack(edges2_all)

    return loc_all, vel_all, change_point_all, edges1_all, edges2_all

def save_file(mode='train'):
    if not osp.exists(args.out):
        os.makedirs(args.out)
    files = ["loc", "vel", 'change_point', "edges1", "edges2"]
    for f in files:
        path = osp.join(args.out, "cp_{}_{}_{}.npy".format(f, mode, suffix))
        print("save file to ", path)
        np.save(path, eval("{}_{}".format(f, mode)))
        
print("Generating {} training simulations".format(args.num_train))
loc_train, vel_train, change_point_train, edges1_train, edges2_train = generate_dataset(args.num_train,
                                                     args.length,
                                                     args.sample_freq)
save_file('train')

print("Generating {} validation simulations".format(args.num_valid))
loc_valid, vel_valid, change_point_valid, edges1_valid, edges2_valid = generate_dataset(args.num_valid,
                                                     args.length,
                                                     args.sample_freq)
save_file('valid')

print("Generating {} test simulations".format(args.num_test))
loc_test, vel_test, change_point_test, edges1_test, edges2_test = generate_dataset(args.num_test,
                                                  args.length,
                                                  args.sample_freq)
save_file('test')
