import torch
import os
import shutil

import pickle
import fairseq
import argparse
import numpy as np
import logging
from time import time, sleep
from tqdm import tqdm

from restorant_dataset import get_dataset
from model import create_model
from torchtext import data

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from utils import AccuracyTensorboradWriter, write_weight_statitsics, ask_user_confirmation

torch.manual_seed(1)

# dataset was downloaded using the link from here:      https://github.com/zhangxiangxiao/Crepe

nsamples = 300000


def checkpoint(model, path):
    if os.path.exists(path):
        os.remove(path)
    torch.save(model.state_dict(), path)


def load_checkpoint(path, *args):
    print(f'loading checkpoint {path}')
    model = create_model(*args)
    model.load_state_dict(torch.load(path))
    return model


def shift_left(tensor, padding_value, device):
    """
    tensor is 2-d, (sequence, batch)
    we are shifting the sequence and we will get (sequence+1, batch)
    """
    assert len(tensor.size()) == 2
    new_tensor = torch.full(tensor.size(), padding_value, dtype=torch.int64, device=device)
    new_tensor[0:-1, :] = tensor[1:, :]
    return new_tensor


def args_to_comment(args):
    comment = ''
    for p in ['note', 'dim', 'ntokens', 'nconv', 'nsamples', 'content_noise']:
        comment += str(p) + '_' + str(args.__dict__[p]) + '_'
    comment = comment[0:-1]
    return comment  # .replace('-', '')


def configure_logger(path):
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Logger created! log path is \n{}'.format(path))
    return logger


def vocab_to_dictionary(vocab):
    decoder_dictionary = fairseq.data.dictionary.Dictionary()
    for i in range(len(vocab)):
        decoder_dictionary.add_symbol(vocab.itos[i])
    return decoder_dictionary


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train lord-seq2seq-convnet.')
    # session
    parser.add_argument('--note', type=str, help='a comment',  required=True)
    parser.add_argument('--device', type=str, default='cpu', help='cuda device: cuda:0 / cuda:1')
    parser.add_argument('--double', action='store_true', help='use only double percision')
    parser.add_argument('--overwrite', action='store_true', help='delete old ckpt with this configuration')
    parser.add_argument('--resume', action='store_true', help='resume training from  old ckpt with this configuration')
    parser.add_argument('--ckpt_every', type=int, default=25, help='how many epochs between checkpoints')
    parser.add_argument('--dir', type=str,
                        default='/root/text_lord/train',
                        help='here the script will create a directory named by the parameters')
    # training
    parser.add_argument('--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--epochs', type=int, help='number pf epochs to train',  required=True)
    parser.add_argument('--content_lr', type=float, help='learning rate for the content embedding',  required=True)
    parser.add_argument('--content_wdecay', type=float, help='weight decay for the content embedding',  required=True)
    # model
    parser.add_argument('--dim', type=int, help='model dimension', required=True)
    parser.add_argument('--content_noise', type=float, help='standard deviation for the content embedding noise',  required=True)
    parser.add_argument('--nsamples', type=int, default=300000,
                        help='number of examples to use')
    parser.add_argument('--ntokens', type=int, default=5,
                        help='number of latent input vectors')
    parser.add_argument('--nconv', type=int, default=20,
                        help='number of conv layers, i think :) default as in the original.')
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
    dataset, vocab = get_dataset(args.nsamples, vocab)
    logger.info(f'dataset loaded, vocab size is {len(vocab)}')

    # serialize the vocab object

    if not args.resume:
        with open(vocab_path, "wb") as file:
            pickle.dump(vocab, file)
            logger.info(f'vocab was pickled into {vocab_path}')

    # the dictionary is used for decoder construction but will never be in use after that.
    decoder_dictionary = vocab_to_dictionary(vocab)

    # build model
    if not args.resume:
        model = create_model(device, args.nsamples, decoder_dictionary.pad(),
                             args.ntokens, args.dim, args.content_noise, 0.1,
                             decoder_dictionary, 50, args.nconv)
    else:
        model = load_checkpoint(os.path.join(foldername, 'last_checkpoint.ckpt'),
                                device, args.nsamples, decoder_dictionary.pad(),
                                args.ntokens, args.dim, args.content_noise, 0.1,
                                decoder_dictionary, 50, args.nconv)


    # training configuration
    model_parameters = [p for p in model.decoder.parameters()] + \
                       [p for p in model.encoder.negative_embedding.parameters()] + \
                       [p for p in model.encoder.positive_embedding.parameters()]

    writer = SummaryWriter(log_dir=foldername, comment=args_to_comment(args))

    logger.info('before entering the training loop, '
                'we are going to have {} iterations per epoch'.format(nsamples // args.batch_size))

    prev_epoch_loss = np.inf
    lr = 0.25
    content_lr = args.content_lr

    acc_writer = AccuracyTensorboradWriter(writer, logger)

    global_step = 0
    for epoch in range(args.epochs):

        logger.info(f'starting epoch {epoch}, previous epoch loss is {prev_epoch_loss}')
        writer.add_scalar('lr/Learning-rate', lr, epoch)
        writer.add_scalar('lr/Content-Learning-rate', content_lr, epoch)

        # optimizers are created each epoch because from time to time there lr is reduced.
        optimizer_model = optim.SGD(model_parameters, lr=lr, momentum=0.99,
                                    nesterov=True)  # parameters convseq2seq paper

        # sparse gradients, no momentum, large lr.
        optimizer_content = optim.SGD(model.parameters(), lr=content_lr,
                                      momentum=0, weight_decay=args.content_wdecay)

        losses = []

        train_iter = data.BucketIterator(dataset=dataset, batch_size=args.batch_size,
                                         sort_key=lambda x: len(x.review), sort=False,
                                         sort_within_batch=True, repeat=False, device=device)

        t = time()
        for batch in tqdm(train_iter):

            batch_size = batch.id.size()[0]

            # create input
            reviews = batch.review.transpose(1, 0)
            src_tokens = torch.cat([batch.id.unsqueeze(1), batch.stars.unsqueeze(1)], dim=1)
            src_lengths = torch.full((batch_size, 1), 5).to(device)

            # run!
            model.zero_grad()
            logits, _ = model(src_tokens, src_lengths, reviews)
            # predictions = F.softmax(logits, dim=2)

            logits_flat = logits.view(-1, len(vocab))
            targets_flat = shift_left(reviews, decoder_dictionary.pad(), device).reshape(-1)#.to(device)

            loss = F.cross_entropy(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.1)
            optimizer_model.step()
            optimizer_content.step()

            # finished training step, now logging
            losses.append(loss.item())
            writer.add_scalar('Loss/per-step', loss.item(), global_step)

            # acc
            if global_step % 200 == 0:
                acc_writer.write_step(logits_flat, targets_flat, global_step)

            writer.add_scalar('Time/step-per-second', 1 / (time() - t), global_step)
            t = time()
            global_step += 1

        writer.add_scalar('Loss/per-epoch', np.average(losses), epoch)
        acc_writer.write_epoch(epoch)

        checkpoint(model, os.path.join(foldername, 'last_checkpoint.ckpt'))
        if epoch % 100 == 0:
            checkpoint(model, os.path.join(foldername, f'epoch{epoch}_checkpoint.ckpt'))

        write_weight_statitsics(writer, model.encoder, epoch)

        # reduce learning rate if needed
        epoch_loss = np.average(losses)
        if epoch_loss >= prev_epoch_loss:
            # if lr > content_lr:
            logger.info(f'epoch {epoch} - reducing learning rate from {lr} to {lr / 3}')
            lr = lr / 3
            if lr < 1e-4:
                logger.info(f'learning rate too low: {lr}. finised.')
                raise Exception()
            # else:
            #     logger.info(f'epoch {epoch} - reducing content learning rate from {content_lr} to {content_lr / 3}')
            #     content_lr = content_lr / 3

        prev_epoch_loss = epoch_loss


if __name__ == '__main__':
    main()









