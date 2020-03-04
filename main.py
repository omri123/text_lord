import torch
import os
import fairseq
import argparse
import numpy as np
import logging
from time import time, sleep

from restorant_dataset import get_dataset
from model import NoEncoder, NoEncoderFConvDecoderModel

from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel

from fairseq.models.fconv import (
    Embedding,
    PositionalEmbedding,
    FConvDecoder
)
from torch.distributions import Normal


from restorant_dataset import RestDataset, lines_generator
from torchtext import data

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(1)

# dataset was downloaded using the link from here:      https://github.com/zhangxiangxiao/Crepe

nsamples = 300000








def checkpoint(model, path):
    if os.path.exists(path):
        os.remove(path)
    torch.save(model.state_dict(), path)


def shift_left(tensor, padding_value):
    """
    tensor is 2-d, (sequence, batch)
    we are shifting the sequence and we will get (sequence+1, batch)
    """
    assert len(tensor.size()) == 2
    new_tensor = torch.full(tensor.size(), padding_value, dtype=torch.int64)
    new_tensor[0:-1, :] = tensor[1:, :]
    return new_tensor


def args_to_comment(args):
    comment = ''
    for p in args.__dict__:
        if '/' not in str(args.__dict__[p]):
            comment += str(p) + ' ' + str(args.__dict__[p]) + ' '
    return comment  # .replace('-', '')


def configure_logger(path):
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('Logger created!')
    print('Logger created! log path is \n{}'.format(path))
    return logger


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train lord-seq2seq-convnet.')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--dim', type=int, help='model dimension')
    parser.add_argument('--epochs', type=int, help='number pf epochs to train')
    parser.add_argument('--content_lr', type=float, help='learning rate for the content embedding')
    parser.add_argument('--nsamples', type=int, default=300000,
                        help='number of examples to use')
    parser.add_argument('--ntokens', type=int, default=5,
                        help='number of latent input vectors')
    parser.add_argument('--nconv', type=int, default=20,
                        help='number of conv layers, i think :) default as in the original.')
    parser.add_argument('--dir', type=str,
                        default='/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train_fixed_shift',
                        help='here the script will create a directory named by the parameters')
    args = parser.parse_args()

    # create directory for checkpoints and logs
    foldername = os.path.join(args.dir, args_to_comment(args))
    print(foldername)
    if os.path.exists(foldername):
        print('you had already tried this configuration! aborting. TODO - resume.')
        sleep(3)
        exit(1)

    os.mkdir(foldername)

    # configure logger
    logger = configure_logger(os.path.join(foldername, 'trainer.log'))

    # create dataset
    dataset, vocab = get_dataset(args.nsamples)
    logger.info(f'dataset loaded, vocab size is {len(vocab)}')
    # serialize the vocab object
    vocab_path = os.path.join(foldername, 'vocab.pickle')
    with open(vocab_path, "wb") as file:
        import pickle
        pickle.dump(vocab, file)
        logger.info(f'vocab was pickled into {vocab_path}')

    # the dictionary is used for decoder construction but will never be in use after that.
    decoder_dictionary = fairseq.data.dictionary.Dictionary()
    for global_step in range(len(vocab)):
        decoder_dictionary.add_symbol(vocab.itos[global_step])

    # build model
    encoder = NoEncoder(sample_size=args.nsamples,
                        padding_index=decoder_dictionary.pad(),
                        ntokens=5,
                        embed_dim=args.dim,
                        noise_std=1.0,
                        dropout=0.1)

    decoder = FConvDecoder(decoder_dictionary,
                           embed_dim=args.dim,
                           out_embed_dim=args.dim,
                           max_positions=50,
                           convolutions=((args.dim, 3),) * args.nconv)

    model = NoEncoderFConvDecoderModel(encoder, decoder).to(device)

    # training configuration
    model_parameters = [p for p in decoder.parameters()] + \
                       [p for p in encoder.negative_embedding.parameters()] + \
                       [p for p in encoder.positive_embedding.parameters()]


    writer = SummaryWriter(log_dir=foldername, comment=args_to_comment(args))

    logger.info('before entering the training loop, '
                'we are going to have {} iterations per epoch'.format(nsamples // args.batch_size))

    prev_epoch_loss = np.inf
    lr = 0.25

    global_step = 0
    for epoch in range(args.epochs):

        # optimizers are created each epoch because from time to time there lr is reduced.
        optimizer_model = optim.SGD(model_parameters, lr=lr, momentum=0.99, nesterov=True)  # parameters convseq2seq paper
        optimizer_content = optim.SGD(model.parameters(), lr=args.content_lr, momentum=0)  # sparse gradients, no momentum, large lr.

        losses = []
        accuracies = []

        train_iter = data.BucketIterator(dataset=dataset, batch_size=args.batch_size,
                                         sort_key=lambda x: len(x.review), sort=False,
                                         sort_within_batch=True, repeat=False)

        t = time()
        for batch in train_iter:
            batch_size = batch.id.size()[0]

            # create input
            ids = batch.id.to(device)
            stars = batch.stars.to(device)
            reviews = batch.review.transpose(1, 0).to(device)

            src_tokens = torch.zeros(batch_size, 2, dtype=torch.int64).to(device)
            src_tokens[:, 0] = ids
            src_tokens[:, 1] = stars

            src_lengths = torch.full((batch_size,1), 5).to(device)

            # run!
            model.zero_grad()
            logits, _ = model(src_tokens, src_lengths, reviews)
            # predictions = F.softmax(logits, dim=2)

            logits_flat = logits.view(-1, len(vocab)) # on cuda
            # targets_flat = reviews.reshape(-1).to(device)
            targets_flat = shift_left(reviews, decoder_dictionary.pad()).reshape(-1).to(device)

            loss = F.cross_entropy(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.1)
            optimizer_model.step()
            optimizer_content.step()

            # finished training step, now logging
            losses.append(loss.item())
            writer.add_scalar('Loss/per-step', loss.item(), global_step)

            logits_flat_np = logits_flat.detach().cpu().numpy()
            targets_flat_np = targets_flat.detach().cpu().numpy()
            acc = np.sum(np.argmax(logits_flat_np, axis=1) == targets_flat_np) / targets_flat_np.size
            accuracies.append(acc)
            writer.add_scalar('Acc/per-step', acc, global_step)

            writer.add_scalar('Time/step-per-second', 1 / (time() - t), global_step)
            t = time()
            global_step += 1

        writer.add_scalar('Loss/per-epoch', np.average(losses), epoch)
        writer.add_scalar('Acc/per-epoch', np.average(accuracies), epoch)

        checkpoint(model, os.path.join(foldername, 'last_checkpoint.ckpt'))
        if epoch % 25 == 0:
            checkpoint(model, os.path.join(foldername, f'epoch{epoch}_checkpoint.ckpt'))

        # log the weights-norm for the parameters of the model
        for weight_name in encoder.state_dict().keys():
            norm = encoder.state_dict()[weight_name].norm().item()
            writer.add_scalar(f'Norm/{weight_name}', norm, epoch)

        # reduce learning rate if needed
        epoch_loss = np.average(losses)
        if epoch_loss >= prev_epoch_loss:
            logger.info(f'epoch {epoch} - reducing learning rate from {lr} to {lr/10}')
            lr = lr / 10



if __name__ == '__main__':
    main()










