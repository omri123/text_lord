import torchtext
from restorant_dataset import START, END
from model import NoEncoderFConvDecoderModel, create_model
from utils import load_checkpoint, vocab_to_dictionary
from restorant_dataset import get_dataset

import torch
import os
import pickle
import numpy as np
from torchtext import data
import copy

from torch.utils.tensorboard import SummaryWriter


def gready_decode_single(model: NoEncoderFConvDecoderModel, vocab: torchtext.vocab,
                         stars: int, sample_id: int, start_token=START, end_token=END):

    max_length = 25

    src_tokens = torch.tensor([[sample_id, stars]], dtype=torch.int64)
    src_lengths = torch.full((1, 1), 5)
    reviews = torch.tensor([[vocab.stoi[start_token]]], dtype=torch.int64)

    sentence = [start_token]

    length = 0
    while end_token not in sentence:
        logits, _ = model(src_tokens, src_lengths, reviews)
        logits_for_new_token = logits[0, -1, :]  # batch, seq, vocab
        word_index = torch.argmax(logits_for_new_token).item()
        sentence.append(vocab.itos[word_index])
        length += 1
        if length > max_length:
            break

        sentence_by_indecies = [vocab.stoi[word] for word in sentence]
        reviews = torch.tensor([sentence_by_indecies], dtype=torch.int64)

    return ' '.join(sentence)




def gready_decode(model, vocab, src_tokens, src_lengths, start_token, end_token):
    pass


import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BeamSearchNode(object):
    def __init__(self, sentence, logProb, length):
        '''
        :param sentence: a list of tokens!!!
        :param logProb: logp sum
        :param length:
        '''
        self.sent = sentence
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return - (self.logp / float(self.leng - 1 + 1e-6) + alpha * reward)

    def __lt__(self, other):
        return self.eval() <= other.eval()


def beam_decode_single(model, vocab, sample_id, stars, beam_width=10, topk=1, SOS_token='<s>', EOS_token='</s>', MAX_LENGTH=50):
    """
    decode single example using beam search.
    :param decoder: a NoEncoderFConvDecoderModel object
    :param vocab:
    :param sample_id:
    :param stars:
    :param beam_width:
    :param SOS_token:
    :param EOS_token:
    :param MAX_LENGTH:
    :return:
    """

    src_tokens = torch.tensor([[sample_id, stars]], dtype=torch.int64)
    src_lengths = torch.full((1, 1), 5)
    review = [SOS_token]
    # review = torch.tensor([[vocab.stoi[SOS_token]]], dtype=torch.int64)

    solutions = []
    nodes = PriorityQueue()
    node = BeamSearchNode(review, 0, 1)
    nodes.put(node)

    qsize = 1
    while True:

        # finished
        if len(solutions) == topk: break

        # give up when decoding takes too long
        if qsize > 2000:
            for i in range(topk - len(solutions)):
                solutions.append(nodes.get())
            break

        # fetch the best node
        node = nodes.get()

        review = node.sent
        review_int = [vocab.stoi[w] for w in review]
        review_torch = torch.tensor([review_int], dtype=torch.int64)

        if review[-1] == EOS_token:
            solutions.append(node)
            continue

        logits, _ = model(src_tokens, src_lengths, review_torch)
        predictions = torch.log_softmax(logits, dim=1)
        log_probs, indexes = torch.topk(predictions, beam_width)

        for new_k in range(beam_width):
            word_index = indexes[0, len(review)-1, new_k].item()
            word = vocab.itos[word_index]
            review_new = copy.deepcopy(node.sent)
            review_new.append(word)

            new_word_log_p = log_probs[0, 0, new_k].item()
            new_node = BeamSearchNode(review_new, node.logp + new_word_log_p, node.leng)
            nodes.put(new_node)

            qsize += 1

    return [' '.join(s.sent) for s in solutions]


def main():
    foldername = '/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/note_tiny_no_noise_dim_32_ntokens_5_nconv_10_nsamples_102400_content_noise_0.001/'
    # foldername = '/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/note_EM_no_noise_dim_32_ntokens_10_nconv_4_nsamples_1024_content_noise_0.0/'
    vocab_path = os.path.join(foldername, 'vocab.pickle')
    model_ckpt_path = os.path.join(foldername, 'last_checkpoint.ckpt')

    with open(vocab_path, 'rb') as file:
        vocab = pickle.load(file)
        print('vocab was loaded')

    decoder_dictionary = vocab_to_dictionary(vocab)


    device = 'cpu'
    nsamples = 102400
    ntokens = 5
    dim = 32
    content_noise = 0.001
    dropout = 0
    nconv = 10

    model = load_checkpoint(model_ckpt_path, 'cpu',
                            device, nsamples, decoder_dictionary.pad(),
                            ntokens, dim, content_noise, dropout,
                            decoder_dictionary, 50, nconv)

    print('model loaded')

    model.eval()

    dataset, vocab = get_dataset(10000, '/cs/labs/dshahaf/omribloch/data/text_lord/restorant/', vocab)

    for i in range(10):
        sid = dataset[i].id
        stars = dataset[i].stars
        # stars = 1
        review_sentence = ' '.join(dataset[i].review)
        print(review_sentence)
        decoded_sentence = gready_decode_single(model, vocab, stars, sid)
        print(decoded_sentence)
        decoded_sentence = gready_decode_single(model, vocab, 1-stars, sid)
        print(decoded_sentence)
        print('-------------')

        decoded_sentence = beam_decode_single(model, vocab, sid, stars, topk=10, beam_width=4)
        for d in decoded_sentence:
            print(d)
        print('==============================')



if __name__=='__main__':
    main()
