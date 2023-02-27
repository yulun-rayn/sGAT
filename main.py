import os
import yaml
import logging
import argparse

import time
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sgat.utils.general_utils import initialize_logger, close_logger
from sgat.utils.math_utils import score_weights, exp_weighted_mse

from sgat.dataset import create_datasets, parse_data_path, parse_data, my_collate
from sgat.dataset.preprocess import read_data

from sgat.model import sGAT

from sgat.train import train, load_current_model, load_best_model

#############################################
#                   ARGS                    #
#############################################

def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--data_path', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')
    add_arg('--gpu', type=int, default=0)
    add_arg('--use_cpu', action='store_true')
    add_arg('--upsample', default=False)
    add_arg('--exp_loss', default=False)
    add_arg('--epoch', type=int, default=1000)
    add_arg('--batch_size', type=int, default=128)
    add_arg('--hidden', type=int, default=256)
    add_arg('--layers', type=int, default=3)
    add_arg('--lr', type=float, default=1e-3)
    add_arg('--workers', type=int, default=12)
    add_arg('--use_3d', action='store_true')
    add_arg('--store_preprocessed', action='store_true')

    return parser.parse_args()


class ArgumentHandler:
    def __init__(self, experiment_dir, starting_lr):
        self.arg_file = os.path.join(experiment_dir, 'args.yaml')
        try:
            self.load_args()
            logging.info("Arguments loaded.")
        except Exception as e:
            self.initialize_args(starting_lr)
            logging.info("Arguments initialized.")

    def load_args(self):
        with open(self.arg_file, 'r') as f:
            self.args = yaml.load(f, Loader=yaml.FullLoader)

    def initialize_args(self, starting_lr):
        args = {}
        args['current_epoch'] = 0
        args['current_lr'] = starting_lr
        args['best_loss'] = 10 ** 10
        self.args = args
        self.save_args()

    def save_args(self):
        with open(self.arg_file, 'w') as f:
            yaml.dump(self.args, f)

    def update_args(self, current_lr, current_epoch, best_loss):
        self.args['current_lr'] = current_lr
        self.args['current_epoch'] = current_epoch
        self.args['best_loss'] = best_loss
        self.save_args()

    def __call__(self, param):
        return self.args[param]


#############################################
#                   MAIN                    #
#############################################

def main(artifact_path,
         score,
         smiles,
         gpu_num=0,
         upsample=False,
         exp_loss=False,
         use_3d=False,
         epoch=1000,
         batch_size=128,
         nb_hidden=256,
         nb_layers=4,
         lr=0.001,
         num_workers=12,
         store_preprocessed=False,
         data_path=None):
    # Global variables: GPU Device, random splits for upsampling, loc and scale parameter for exp weighted loss.
    global exp_loc
    global exp_scale

    device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")

    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(artifact_path, 'runs/' + dt))
    save_dir = os.path.join(artifact_path, 'saves/' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)

    arg_handler = ArgumentHandler(save_dir, lr)

    train_data, valid_data, test_data = create_datasets(score, smiles, use_3d)
    valid_data.compute_baseline_error()
    print("Dataset created")

    if (data_path is not None) and store_preprocessed:
        print("Using stored dataset. Preprocessing if necessary.")
        storage_path = parse_data_path(data_path, use_3d)
        train_data = parse_data(train_data, storage_path, 'train')
        valid_data = parse_data(valid_data, storage_path, 'valid')
        #test_data  = parse_data(test_data, storage_path, 'test')

    if upsample:
        # Percentiles used in score weights.
        # Reset randomness
        np.random.seed()
        #train_25 = np.percentile(train_data.score, 25)
        #train_75 = np.percentile(train_data.score, 75)
        upsampled_weight = np.random.uniform(0.5, 1, 1)[0]
        #split = np.random.uniform(train_25, train_75, 1)[0]
        split = np.percentile(train_data.score, 1)
        logging.info("Upsampling weights: {:3.2f}".format(upsampled_weight))
        logging.info("Upsampling split: {:3.2f}".format(split))

        # Initialize weighted sampler
        train_weights = torch.DoubleTensor(score_weights(train_data.score, split, upsampled_weight))
        valid_weights = torch.DoubleTensor(score_weights(valid_data.score, split, upsampled_weight))
        #test_weights = torch.DoubleTensor(score_weights(test_data.score, split, upsampled_weight))

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(valid_weights, len(valid_weights))
        #test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))

        train_loader = DataLoader(train_data,
                                  collate_fn=my_collate,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers)
        valid_loader = DataLoader(valid_data,
                                  collate_fn=my_collate,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=num_workers)
    else:
        train_loader = DataLoader(train_data,
                                  shuffle=True,
                                  collate_fn=my_collate,
                                  batch_size=batch_size,
                                  num_workers=num_workers)
        valid_loader = DataLoader(valid_data,
                                  collate_fn=my_collate,
                                  batch_size=batch_size,
                                  num_workers=num_workers)


    try:
        net = load_current_model(save_dir)
        logging.info("Model restored")
    except Exception as e:
        input_dim, nb_edge_types = train_data.get_graph_spec()
        net = sGAT(input_dim=input_dim,
                        nb_hidden=nb_hidden,
                        nb_layers=nb_layers,
                        nb_edge_types=nb_edge_types,
                        use_3d=use_3d)
        logging.info(net)
        logging.info("New model created")
    net.to_device(device)

    optim = torch.optim.Adam(net.parameters(), lr=arg_handler('current_lr'))

    if exp_loss:
        np.random.seed()
        exp_loc = min(train_data.score)
        exp_scale = np.random.uniform(1, 4, 1)[0]
        logging.info("Exponential loc: {:3.2f}".format(exp_loc))
        logging.info("Exponential scale: {:3.2f}".format(exp_scale))
        def loss_criterion(output, target):
            return exp_weighted_mse(output, target, exp_loc, exp_scale)
        criterion = loss_criterion
    else:
        criterion = torch.nn.MSELoss()

    train(net,
          criterion,
          epoch,
          batch_size,
          train_loader,
          valid_loader,
          optim,
          arg_handler,
          save_dir,
          writer)

    close_logger()
    writer.close()
    return load_best_model(save_dir)


if __name__ == "__main__":
    args = read_args()

    artifact_path = os.path.join(args.artifact_path, args.name)
    os.makedirs(artifact_path, exist_ok=True)

    scores, smiles = read_data(args.data_path)

    main(artifact_path,
         scores,
         smiles,
         gpu_num=args.gpu,
         upsample=args.upsample,
         exp_loss=args.exp_loss,
         use_3d=args.use_3d,
         epoch=args.epoch,
         batch_size=args.batch_size,
         nb_hidden=args.hidden,
         nb_layers=args.layers,
         lr=args.lr,
         num_workers=args.workers,
         store_preprocessed=args.store_preprocessed,
         data_path = args.data_path)
