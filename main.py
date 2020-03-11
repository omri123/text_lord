import torch
import sys
import shutil

import pickle
import argparse
import logging

from restorant_dataset import get_dataset
from model import create_model
import os
from torch.utils.tensorboard import SummaryWriter
from utils import AccuracyTensorboradWriter, write_weight_statitsics, \
    ask_user_confirmation, load_checkpoint, checkpoint, vocab_to_dictionary

from eval import Evaluator
from train import train
from eval import evaluate
torch.manual_seed(1)

# dataset was downloaded using the link from here:      https://github.com/zhangxiangxiao/Crepe

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
    #
    # fh = logging.FileHandler(path)
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    # create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Logger created! log path is \n{}'.format(path))
    return logger


def folder_setup(overwrite, resume, foldername, force):
    if overwrite and resume:
        raise Exception("can't use overwrite and resume together!!!")

    if os.path.exists(foldername):
        if overwrite:
            if force or ask_user_confirmation('overwriting'):
                shutil.rmtree(foldername)
            else:
                print('okey, exiting. not removing anything.')
                exit(0)
        elif resume:
            if force or ask_user_confirmation('resuming'):
                print("resuming!")
            else:
                print('okey, exiting. not resuming.')
                exit(0)
        else:
            raise Exception('you had already tried this configuration! aborting. try --overwrite or --resume.')

    if not os.path.exists(foldername):
        os.makedirs(foldername)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train lord-seq2seq-convnet.')
    # session
    parser.add_argument('--note', type=str, help='a comment',  required=True)
    parser.add_argument('--device', type=str, default='cpu', help='cuda device: cuda:0 / cuda:1')
    parser.add_argument('--overwrite', action='store_true', help='delete old ckpt with this configuration')
    parser.add_argument('--resume', action='store_true', help='resume training from  old ckpt with this configuration')
    parser.add_argument('--shuffle', action='store_true', help='shuffle input')
    parser.add_argument('--ckpt_every', type=int, default=25, help='how many epochs between checkpoints')
    parser.add_argument('--dir', type=str,
                        default='/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train',
                        help='here the script will create a directory named by the parameters')
    parser.add_argument('--data_dir', type=str, default='/cs/labs/dshahaf/omribloch/data/text_lord/restorant/')
    parser.add_argument('-f', action='store_true')
    # training
    parser.add_argument('--batch_size', type=int, help='batch size', required=True)
    parser.add_argument('--epochs', type=int, help='number pf epochs to train',  required=True)
    parser.add_argument('--it', type=int, help='number pf train-eval iterations',  required=True)
    parser.add_argument('--content_wdecay', type=float, help='weight decay for the content embedding',  required=True)
    parser.add_argument('--drop_connect', type=float, help='drop connect rate', default=0)
    # model
    parser.add_argument('--dim', type=int, help='model dimension', required=True)
    parser.add_argument('--content_noise', type=float, help='standard deviation for the content embedding noise',  required=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='embedding dropout')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--nsamples', type=int, default=300000,
                        help='number of examples to use')
    parser.add_argument('--ntokens', type=int, default=5,
                        help='number of latent input vectors')
    parser.add_argument('--nconv', type=int, default=20,
                        help='number of conv layers, i think :) default as in the original.')
    args = parser.parse_args()



    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    # create directory for checkpoints and logs
    foldername = os.path.join(args.dir, args_to_comment(args))
    print(foldername)

    folder_setup(args.overwrite, args.resume, foldername, force=args.f)

    # configure logger
    logger = configure_logger(os.path.join(foldername, 'trainer.log'))

    vocab_path = os.path.join(foldername, 'vocab.pickle')
    vocab = None
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
                             args.ntokens, args.dim, args.content_noise, args.dropout,
                             decoder_dictionary, 50, args.nconv)
    else:
        model = load_checkpoint(os.path.join(foldername, 'last_checkpoint.ckpt'), device,
                                device, args.nsamples, decoder_dictionary.pad(),
                                args.ntokens, args.dim, args.content_noise, 0.1,
                                decoder_dictionary, 50, args.nconv)

    writer = SummaryWriter(log_dir=foldername, comment=args_to_comment(args))

    global_step = 0
    global_epoch = 0
    for it in range(args.it):
        logger.info('-- iteration {} --'.format(it))
        global_step, global_epoch = train(model, dataset, device, args.epochs, args.batch_size, decoder_dictionary.pad(), logger, args.content_wdecay,
              writer, foldername, global_step=global_step, global_epoch=global_epoch, shuffle=args.shuffle)
        evaluate(model, vocab, dataset, 1000, it, logger, writer, device=device, gready=False)
    print('finished')



if __name__ == '__main__':
    main()









