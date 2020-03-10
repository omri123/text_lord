from utils import load_checkpoint, vocab_to_dictionary
from restorant_dataset import get_dataset
import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm

from main import PPLEvaluator
import fasttext

labels_dictionary = {'__label__positive': 1, '__label__negative': 0}


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='evaluate lord-seq2seq-convnet.')
    # session
    parser.add_argument('--foldername', type=str)
    parser.add_argument('--data_dir', type=str, default='/cs/labs/dshahaf/omribloch/data/text_lord/restorant/')
    parser.add_argument('--ckpt_name', type=str, default='last_checkpoint.ckpt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--nsamples', type=int)
    parser.add_argument('--ntokens', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--content_noise', type=float)
    parser.add_argument('--nconv', type=int)
    parser.add_argument('--samples_to_eval', type=int)
    parser.add_argument('--gready', action='store_true')


    args = parser.parse_args()

    vocab_path = os.path.join(args.foldername, 'vocab.pickle')
    model_ckpt_path = os.path.join(args.foldername, 'last_checkpoint.ckpt')

    with open(vocab_path, 'rb') as file:
        vocab = pickle.load(file)
        print('vocab was loaded')

    decoder_dictionary = vocab_to_dictionary(vocab)
    dropout = 0

    model = load_checkpoint(model_ckpt_path, args.device,
                            args.device, args.nsamples, decoder_dictionary.pad(),
                            args.ntokens, args.dim, args.content_noise, dropout,
                            decoder_dictionary, 50, args.nconv)

    print('model loaded')

    model.eval()

    dataset, vocab = get_dataset(args.nsamples, args.data_dir, vocab)

    evaluator = PPLEvaluator()
    fasttext_classfier = fasttext.FastText.load_model('/cs/labs/dshahaf/omribloch/data/text_lord/restorant/fasttext_model.bin')

    orig_ppl = []
    orig_bleu = []

    new_ppl = []
    new_bleu = []

    correct_counter = 0

    for i in tqdm(range(args.samples_to_eval)):
        sid = dataset[i].id
        stars = dataset[i].stars
        # stars = 1
        review_sentence = ' '.join(dataset[i].review)

        if args.gready:
            ppl, bleu, _ = evaluator.eval(model, vocab, review_sentence, stars, sid)
            orig_ppl.append(ppl)
            orig_bleu.append(bleu)

            ppl, bleu, s = evaluator.eval(model, vocab, review_sentence, 1-stars, sid)
            new_ppl.append(ppl)
            new_bleu.append(bleu)

            predicted_label = fasttext_classfier.predict(s)[0][0]
            if labels_dictionary[predicted_label] == 1-stars:
                correct_counter += 1

        else:
            raise Exception('not implemented yet!')

    print(f'orig ppl: {np.average(orig_ppl)}')
    print(f'new ppl: {np.average(new_ppl)}')

    print(f'orig bleu: {np.average(orig_bleu)}')
    print(f'new bleu: {np.average(new_bleu)}')

    print(f'classifier accuracy: {correct_counter / args.samples_to_eval}')


print('haha')

if __name__ == '__main__':
    print('hoho')
    main()

