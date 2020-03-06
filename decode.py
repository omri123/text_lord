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


def main():

    foldername = '/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/note_no_noise_dim_32_ntokens_5_nconv_20_nsamples_100_content_noise_0.0/'
    foldername = '/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/note_no_noise_dim_128_ntokens_5_nconv_20_nsamples_100_content_noise_0.0/'
    vocab_path = os.path.join(foldername, 'vocab.pickle')
    model_ckpt_path = os.path.join(foldername, 'last_checkpoint.ckpt')

    with open(vocab_path, 'rb') as file:
        vocab = pickle.load(file)
        print('vocab was loaded')

    decoder_dictionary = vocab_to_dictionary(vocab)


    device = 'cpu'
    nsamples = 100
    ntokens = 5
    dim = 128
    content_noise = 0.0
    dropout = 0
    nconv = 20

    model = load_checkpoint(model_ckpt_path, 'cpu',
                            device, nsamples, decoder_dictionary.pad(),
                            ntokens, dim, content_noise, dropout,
                            decoder_dictionary, 50, nconv)

    print('model loaded')

    model.eval()

    dataset, vocab = get_dataset(10000, '/cs/labs/dshahaf/omribloch/data/text_lord/restorant/', vocab)

    for i in [0, 10, 90]:
        sid = dataset[i].id
        stars = dataset[i].stars
        review_sentence = ' '.join(dataset[i].review)
        print(review_sentence)
        decoded_sentence = gready_decode_single(model, vocab, stars, sid)
        print(decoded_sentence)
        print('-------------')



if __name__=='__main__':
    main()
