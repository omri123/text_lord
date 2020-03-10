import torch
import os
import sys
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
from utils import AccuracyTensorboradWriter, write_weight_statitsics, \
    ask_user_confirmation, load_checkpoint, checkpoint, vocab_to_dictionary

import kenlm
from decode import gready_decode_single
from nltk.translate.bleu_score import sentence_bleu
torch.manual_seed(1)

# dataset was downloaded using the link from here:      https://github.com/zhangxiangxiao/Crepe

nsamples = 300000


class PPLEvaluator:
    # evaluate the pp of a sentence
    def __init__(self):
        self.lm = kenlm.Model('/cs/labs/dshahaf/omribloch/data/text_lord/restorant/kenlm.arpa')

    def eval(self, model, vocab, review, stars: int, sid: int, gready=True):
        sentence_as_list = gready_decode_single(model, vocab, stars, sid).split(' ')[1:-1] # remove <s> and <\s>
        sentence_as_string = ' '.join(sentence_as_list)
        ppl = self.lm.perplexity(sentence_as_string)

        review_as_list = review.split(' ')[1:-1]
        bleu = sentence_bleu([review_as_list], sentence_as_list, weights=[1, 0, 0, 0])

        return ppl, bleu, sentence_as_string

def drop_connect(model, rate):
    with torch.no_grad():
        dropout = torch.nn.Dropout(p=rate, inplace=True)
        for p in model.parameters():
            dropout(p)


def shift_left(tensor, padding_value, device):
    """
    tensor is 2-d, (sequence, batch)
    we are shifting the sequence and we will get (sequence+1, batch)
    """
    assert len(tensor.size()) == 2
    new_tensor = torch.full(tensor.size(), padding_value, dtype=torch.int64, device=device)
    new_tensor[:, 0:-1] = tensor[:, 1:]
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
                        default='/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train',
                        help='here the script will create a directory named by the parameters')
    parser.add_argument('--data_dir', type=str, default='/cs/labs/dshahaf/omribloch/data/text_lord/restorant/')
    # training
    parser.add_argument('--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--epochs', type=int, help='number pf epochs to train',  required=True)
    parser.add_argument('--content_wdecay', type=float, help='weight decay for the content embedding',  required=True)
    parser.add_argument('--drop_connect', type=float, help='drop connect rate', default=0)
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
    dataset, vocab = get_dataset(args.nsamples, args.data_dir, vocab)
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
        model = load_checkpoint(os.path.join(foldername, 'last_checkpoint.ckpt'), device,
                                device, args.nsamples, decoder_dictionary.pad(),
                                args.ntokens, args.dim, args.content_noise, 0.1,
                                decoder_dictionary, 50, args.nconv)

    model.to(device)
    model.train()


    # training configuration
    model_parameters = [p for p in model.decoder.parameters()] + \
                       [p for p in model.encoder.negative_embedding.parameters()] + \
                       [p for p in model.encoder.positive_embedding.parameters()]
    content_parameters = [p for p in model.encoder.content_embeddings.parameters()]


    writer = SummaryWriter(log_dir=foldername, comment=args_to_comment(args))

    logger.info('before entering the training loop, '
                'we are going to have {} iterations per epoch'.format(nsamples // args.batch_size))

    acc_writer = AccuracyTensorboradWriter(writer, logger)
    ppl_evaluator = PPLEvaluator()

    optimizer = optim.Adagrad(model_parameters)
    content_optimizer = optim.Adagrad(content_parameters)

    global_step = 0
    for epoch in range(args.epochs):


        losses = []
        ppl = []

        train_iter = data.BucketIterator(dataset=dataset, batch_size=args.batch_size,
                                         sort_key=lambda x: len(x.review), sort=False,
                                         sort_within_batch=True, repeat=False, device='cpu')

        t = time()
        for step, batch in enumerate(tqdm(train_iter)):

            batch_size = batch.id.size()[0]

            # create input
            reviews = batch.review.transpose(1, 0).to(device)

            src_tokens = torch.cat([batch.id.unsqueeze(1), batch.stars.unsqueeze(1)], dim=1).to(device)
            src_lengths = torch.full((batch_size, 1), 5).to(device)

            # run!
            model.zero_grad()
            logits, _ = model(src_tokens, src_lengths, reviews)

            logits_flat = logits.view(-1, len(vocab))
            targets_flat = shift_left(reviews, decoder_dictionary.pad(), device).reshape(-1)#.to(device)

            embeddings = model.encoder(src_tokens, src_lengths)['encoder_out'][0]
            regloss = embeddings.norm()
            writer.add_scalar('Loss/wdecay', regloss.item(), global_step)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=decoder_dictionary.pad()) + regloss * args.content_wdecay
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()
            content_optimizer.step()

            # finished training step, now logging
            losses.append(loss.item())
            writer.add_scalar('Loss/per-step', loss.item(), global_step)

            # acc
            if global_step % 10 == 0:
                acc_writer.write_step(logits_flat, targets_flat, global_step, ignore_index=decoder_dictionary.pad())
                ppls = []
                bleus = []
                for i in range(batch_size):
                    sid = batch.id[i].item()
                    stars = batch.stars[i].item()
                    ppl, bleu = ppl_evaluator.eval(model, vocab, dataset[step].review, stars, sid)
                    ppls.append(ppl)
                    bleus.append(bleu)

                writer.add_scalar('metrices/PPL-step', np.average(ppls), global_step)
                writer.add_scalar('metrices/BLEU-step', np.average(bleus), global_step)
                ppls.append(np.average(ppls))
                ppls.append(np.average(bleus))


            writer.add_scalar('Time/step-per-second', 1 / (time() - t), global_step)
            t = time()
            global_step += 1

        drop_connect(model, args.drop_connect)
        writer.add_scalar('Loss/per-epoch', np.average(losses), epoch)
        writer.add_scalar('metrices/PPL-epoch', np.average(ppls), epoch)
        writer.add_scalar('metrices/BLEU-epoch', np.average(bleus), epoch)
        logger.info('epoch {} loss {}'.format(epoch, np.average(losses)))
        acc_writer.write_epoch(epoch)

        if epoch % args.ckpt_every == 0:
            checkpoint(model, os.path.join(foldername, f'epoch{epoch}_checkpoint.ckpt'))
            checkpoint(model, os.path.join(foldername, 'last_checkpoint.ckpt'))

        write_weight_statitsics(writer, model.encoder, epoch)


if __name__ == '__main__':
    main()









