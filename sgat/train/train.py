import os
import sys
import time
import logging

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric as pyg

from ..model.model import sGAT, save_sGAT, load_sGAT

torch.multiprocessing.set_sharing_strategy('file_system')

#####################################################
#                   MODEL HANDLING                  #
#####################################################

def load_current_model(model_path):
    net = load_sGAT(os.path.join(model_path, 'current_model.pt'))
    return net

def load_best_model(model_path):
    net = load_sGAT(os.path.join(model_path, 'best_model.pt'))
    return net

def save_current_model(net, model_path):
    save_sGAT(net, os.path.join(model_path, 'current_model.pt'))

def save_best_model(net, model_path):
    save_sGAT(net, os.path.join(model_path, 'best_model.pt'))


#################################################
#                   TRAINING                    #
#################################################

def proc_one_epoch(net,
                   criterion,
                   batch_size,
                   loader,
                   optim=None,
                   train=False):
    print_freq = 10 if train else 4
    nb_batch = len(loader)
    nb_samples = nb_batch * batch_size

    epoch_loss = 0.0
    elapsed = 0.0

    if train:
        net.train()
    else:
        net.eval()

    t0 = time.time()
    logging.info("  {} batches, {} samples".format(nb_batch, nb_samples))
    for i, (y, G1, G2) in enumerate(loader):
        t1 = time.time()
        if train:
            optim.zero_grad()
        y = y.to(net.device, non_blocking=True)
        G1 = G1.to(net.device)
        G2 = G2.to(net.device) if G2 is not None else None

        y_pred = net(G1, G2).squeeze()

        loss = criterion(y_pred, y)
        with torch.autograd.set_detect_anomaly(True):
            if train:
                loss.backward()
                optim.step()
        epoch_loss += loss.item()

        if ((i + 1) % (nb_batch // print_freq)) == 0:
            nb_proc = (i + 1) * batch_size
            logging.info("    {:8d}: {:4.2f}".format(nb_proc, epoch_loss / (i + 1)))
        elapsed += time.time() - t1

    logging.info("  Model elapsed:  {:.2f}".format(elapsed))
    logging.info("  Loader elapsed: {:.2f}".format(time.time() - t0 - elapsed))
    logging.info("  Total elapsed:  {:.2f}".format(time.time() - t0))
    return epoch_loss / nb_batch


def train(net,
          criterion,
          epoch,
          batch_size,
          train_loader,
          valid_loader,
          optim,
          arg_handler,
          save_dir,
          writer):
    current_lr = optim.param_groups[0]['lr']
    lr_end = current_lr / 10 ** 3

    best_loss = arg_handler('best_loss')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=3, verbose=True)
    scheduler.step(best_loss)
    for i in range(arg_handler('current_epoch'), epoch):
        t0 = time.time()
        logging.info("\n\nEpoch {}".format(i + 1))
        logging.info("Learning rate: {0:.3g}".format(current_lr))
        logging.info("  Train:")
        train_loss = proc_one_epoch(net,
                                    criterion,
                                    batch_size,
                                    train_loader,
                                    optim,
                                    train=True)
        logging.info("\n  Valid:")
        valid_loss = proc_one_epoch(net,
                                    criterion,
                                    batch_size,
                                    valid_loader)
        logging.info("Train MSE: {:3.2f}".format(train_loss))
        logging.info("Valid MSE: {:3.2f}".format(valid_loss))
        writer.add_scalar('lr', current_lr, i)
        writer.add_scalar('train_loss', train_loss, i)
        writer.add_scalar('valid_loss', valid_loss, i)
        #writer.add_scalars('loss',
        #                   {'train': train_loss, 'valid': valid_loss},
        #                   i)
        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            logging.info("Best performance on valid set")
            best_loss = valid_loss
            save_best_model(net, save_dir)
        logging.info("{:6.1f} seconds, this epoch".format(time.time() - t0))

        current_lr = optim.param_groups[0]['lr']
        arg_handler.update_args(current_lr, i + 1, best_loss)
        save_current_model(net, save_dir)
        if current_lr < lr_end:
            break
