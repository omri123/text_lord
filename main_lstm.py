import torch
import sys
import shutil

import pickle
import argparse
import numpy as np
import logging
from time import time, sleep
from tqdm import tqdm

from restorant_dataset import get_dataset
from torchtext import data

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from utils import AccuracyTensorboradWriter, write_weight_statitsics, \
    ask_user_confirmation, load_checkpoint, checkpoint

from model_lstmp import LSTM_LORD

np.set_printoptions(precision=3)

torch.manual_seed(1)


def shift_left(tensor, padding_value, device):
    """
    tensor is 2-d, (batch, sequence)
    we are shifting the sequence and we will get (batch, sequence+1)
    """
    assert len(tensor.size()) == 2
    new_tensor = torch.full(tensor.size(), padding_value, dtype=torch.int64, device=device)
    new_tensor[:, 0:-1] = tensor[:, 1:]
    return new_tensor


def args_to_comment(args):
    comment = ''
    for p in ['note', 'dim', 'nlayers', 'nsamples', 'content_noise']:
        comment += str(p) + '_' + str(args.__dict__[p]) + '_'
    comment = comment[0:-1]
    return comment  # .replace('-', '')


def configure_logger(path):
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Logger created! log path is \n{}'.format(path))
    return logger


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train lord-seq2seq-convnet.')
    # session
    parser.add_argument('--note', type=str, help='a comment',  required=True)
    parser.add_argument('--device', type=str, default='cpu', help='cuda device: cuda:0 / cuda:1')
    parser.add_argument('--overwrite', action='store_true', help='delete old ckpt with this configuration')
    parser.add_argument('--resume', action='store_true', help='resume training from  old ckpt with this configuration')
    parser.add_argument('--ckpt_every', type=int, default=25, help='how many epochs between checkpoints')
    parser.add_argument('--dir', type=str,
                        default='/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/lstm',
                        help='here the script will create a directory named by the parameters')
    parser.add_argument('--data_dir', type=str, default='/cs/labs/dshahaf/omribloch/data/text_lord/restorant/')
    # training
    parser.add_argument('--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--epochs', type=int, help='number pf epochs to train',  required=True)
    parser.add_argument('--content_wdecay', type=float, help='weight decay for the content embedding',  required=True)
    # model
    parser.add_argument('--dim', type=int, help='model dimension', required=True)
    parser.add_argument('--content_noise', type=float, help='standard deviation for the content embedding noise',  required=True)
    parser.add_argument('--nsamples', type=int, default=300000, help='number of examples to use')
    parser.add_argument('--nlayers', type=int, default=2, help='number of lstm layers')


    args = parser.parse_args()

    if args.overwrite and args.resume:
        raise Exception("can't use overwrite and resume together!!!")

    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    # create directory for checkpoints and logs
    foldername = os.path.join(args.dir, args_to_comment(args))
    print(foldername)

    vocab = None
    model = None

    if os.path.exists(foldername):
        if args.overwrite:
            if ask_user_confirmation('overwriting'):
                shutil.rmtree(foldername)
            else:
                print('okey, exiting. not removing anything.')
                exit(0)
        elif args.resume:
            if ask_user_confirmation('resuming'):
                print("resuming!")
            else:
                print('okey, exiting. not resuming.')
                exit(0)
        else:
            raise Exception('you had already tried this configuration! aborting. try --overwrite or --resume.')

    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # configure logger
    logger = configure_logger(os.path.join(foldername, 'trainer.log'))

    vocab_path = os.path.join(foldername, 'vocab.pickle')

    if args.resume:
        with open(vocab_path, 'rb') as file:
            vocab = pickle.load(file)
            print('vocab was loaded')

    # create dataset
    dataset, vocab = get_dataset(args.nsamples, args.data_dir, vocab)
    logger.info(f'dataset loaded, vocab size is {len(vocab)}')

    # serialize the vocab object

    if not args.resume:
        with open(vocab_path, "wb") as file:
            pickle.dump(vocab, file)
            logger.info(f'vocab was pickled into {vocab_path}')

    # build model
    if not args.resume:
        model = LSTM_LORD(args.dim, args.nlayers, len(vocab), args.nsamples, args.content_noise)
    else:
        model = load_checkpoint(os.path.join(foldername, 'last_checkpoint.ckpt'), device,
                                args.dim, args.nlayers, len(vocab), args.nsample)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    model.to(device)
    model.train()

    writer = SummaryWriter(log_dir=foldername, comment=args_to_comment(args))

    logger.info('before entering the training loop, '
                'we are going to have {} iterations per epoch'.format(args.nsamples // args.batch_size))

    acc_writer = AccuracyTensorboradWriter(writer, logger)

    global_step = 0
    for epoch in range(args.epochs):

        # optimizers are created each epoch because from time to time there lr is reduced.
        model_parameters = [p for p in model.lstm.parameters()] + \
            [p for p in model.stars_embedding.parameters()] + \
            [p for p in model.fc.parameters()]

        content_parameters = [p for p in model.sample_embedding.parameters()]

        # optimizer = optim.Adam(model_parameters, lr=0.001)
        # content_optimizer = optim.Adam(content_parameters, lr=0.1, weight_decay=args.content_wdecay)
        optimizer = optim.Adagrad(model.parameters())

        losses = []

        train_iter = data.BucketIterator(dataset=dataset, batch_size=args.batch_size,
                                         sort_key=lambda x: len(x.review), sort=False,
                                         sort_within_batch=True, repeat=False, device=device)

        for batch in tqdm(train_iter):

            # create input
            reviews = batch.review.transpose(1, 0)

            state = model.create_initial_hiddens(batch.stars, batch.id)

            # run!
            model.zero_grad()
            logits, state = model(reviews, state)

            logits_flat = logits.view(-1, len(vocab))
            targets_flat = shift_left(reviews, 1, device).reshape(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=1)
            loss.backward()

            optimizer.step()
            # content_optimizer.step()

            # finished training step, now logging
            losses.append(loss.item())
            writer.add_scalar('Loss/per-step', loss.item(), global_step)

            # acc
            if global_step % 1 == 0:
                acc_writer.write_step(logits_flat, targets_flat, global_step, ignore_index=1)

            global_step += 1

        logger.info('epoch {} loss {}'.format(epoch, np.average(losses)))
        writer.add_scalar('Loss/per-epoch', np.average(losses), epoch)
        acc_writer.write_epoch(epoch)

        checkpoint(model, os.path.join(foldername, 'last_checkpoint.ckpt'))
        if epoch % 100 == 0:
            checkpoint(model, os.path.join(foldername, f'epoch{epoch}_checkpoint.ckpt'))


if __name__ == '__main__':
    main()









