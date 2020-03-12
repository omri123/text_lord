from utils import load_checkpoint, vocab_to_dictionary, load_checkpoint_partitioned
from restorant_dataset import get_dataset
import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import fasttext
import kenlm

from decode import gready_decode_single, beam_decode_single
from torch.utils.tensorboard import SummaryWriter

class Evaluator:
    # evaluate the pp of a sentence
    def __init__(self):
        self.lm = kenlm.Model('/cs/labs/dshahaf/omribloch/data/text_lord/restorant/kenlm.arpa')
        self.fasttext_classfier = fasttext.FastText.load_model('/cs/labs/dshahaf/omribloch/data/text_lord/restorant/fasttext_model.bin')
        self.labels_dictionary = {'__label__positive': 1, '__label__negative': 0}

    def evel_sentence(self, generated, original, generated_star):
        ppl = self.lm.perplexity(generated)
        bleu = sentence_bleu([original.split(' ')], generated.split(' '), weights=[1, 0, 0, 0])

        predicted_label = self.fasttext_classfier.predict(generated)[0][0]
        if self.labels_dictionary[predicted_label] == generated_star:
            classified = 1
        else:
            classified = 0


        return ppl, bleu, classified

    def eval(self, model, vocab, review, stars: int, sid: int, gready=True, device='cpu'):
        if gready:
            lgenerated = gready_decode_single(model, vocab, stars, sid, device=device).split(' ')[1:-1] # remove <s> and <\s>
        else:
            lgenerated = beam_decode_single(model, vocab, sid, stars, beam_width=5, topk=1, device=device)[0].split(' ')[1:-1]
        sgenerated = ' '.join(lgenerated)
        ppl = self.lm.perplexity(sgenerated)

        loriginal = review.split(' ')[1:-1]
        soriginal = ' '.join(loriginal)
        original_ppl = self.lm.perplexity(soriginal)
        bleu = sentence_bleu([loriginal], lgenerated, weights=[1, 0, 0, 0])

        predicted_label = self.fasttext_classfier.predict(sgenerated)[0][0]
        if self.labels_dictionary[predicted_label] == stars:
            classified = 1
        else:
            classified = 0

        return ppl, bleu, classified, soriginal, sgenerated, original_ppl


def evaluate(model, vocab, dataset, nsamples_to_eval, iteration_number, logger, writer: SummaryWriter, gready=True, device='cpu'):
    model = model.eval()
    evaluator = Evaluator()

    orig_ppl = []

    reconstruct_ppl =[]
    reconstruct_bleu = []

    new_ppl = []
    new_bleu = []

    counter = 0
    classified_correct_reconstruct = 0
    classified_correct_generated = 0

    for i in tqdm(range(len(dataset))):
        if i % (len(dataset) // nsamples_to_eval) == 0:

            sid = dataset[i].id
            stars = dataset[i].stars
            review_sentence = ' '.join(dataset[i].review)

            ppl, bleu, classified, soriginal, sgenerated, original_ppl = evaluator.eval(model, vocab, review_sentence, stars, sid, gready, device=device)
            orig_ppl.append(original_ppl)
            reconstruct_ppl.append(ppl)
            reconstruct_bleu.append(bleu)
            classified_correct_reconstruct += classified

            ppl, bleu, classified, soriginal, sgenerated_new, original_ppl = evaluator.eval(model, vocab, review_sentence, 1-stars, sid, gready, device=device)
            new_ppl.append(ppl)
            new_bleu.append(bleu)
            classified_correct_generated += 1

            if i % (len(dataset) // 10) == 0: # write only 10 texts to tensorboard.
                writer.add_text(f'sample_{i}_orig', soriginal, iteration_number)
                writer.add_text(f'sample_{i}_reconstruct', sgenerated, iteration_number)
                writer.add_text(f'sample_{i}_new', sgenerated_new, iteration_number)
                writer.add_text(f'sample_{i}_original_sentiment', str(stars), iteration_number)

            counter += 1

    writer.add_scalar('orig_ppl', np.average(orig_ppl), iteration_number)

    writer.add_scalar('reconstruct_ppl', np.average(reconstruct_ppl), iteration_number)
    writer.add_scalar('reconstruct_bleu', np.average(reconstruct_bleu), iteration_number)
    writer.add_scalar('new_ppl', np.average(new_ppl), iteration_number)
    writer.add_scalar('new_bleu', np.average(new_bleu), iteration_number)

    writer.add_scalar('classified_correct_reconstruct', classified_correct_reconstruct / counter, iteration_number)
    writer.add_scalar('classified_correct_generated', classified_correct_generated / counter, iteration_number)

    logger.info('orig_ppl {}'.format(np.average(orig_ppl)))

    logger.info('reconstruct_ppl {}'.format(np.average(reconstruct_ppl)))
    logger.info('reconstruct_bleu {}'.format(np.average(reconstruct_bleu)))
    logger.info('new_ppl {}'.format(np.average(new_ppl)))
    logger.info('new_bleu {}'.format(np.average(new_bleu)))

    logger.info('classified_correct_reconstruct {}'.format(classified_correct_reconstruct / counter))
    logger.info('classified_correct_generated {}'.format(classified_correct_generated / counter))




















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
    parser.add_argument('--partitioned', action='store_true')


    args = parser.parse_args()

    vocab_path = os.path.join(args.foldername, 'vocab.pickle')
    model_ckpt_path = os.path.join(args.foldername, 'last_checkpoint.ckpt')

    with open(vocab_path, 'rb') as file:
        vocab = pickle.load(file)
        print('vocab was loaded')

    decoder_dictionary = vocab_to_dictionary(vocab)
    dropout = 0

    if not args.partitioned:
        model = load_checkpoint(model_ckpt_path, args.device,
                                args.device, args.nsamples, decoder_dictionary.pad(),
                                args.ntokens, args.dim, args.content_noise, dropout,
                                decoder_dictionary, 50, args.nconv)
    else:
        model = load_checkpoint_partitioned(model_ckpt_path, args.device,
                                args.device, args.nsamples, decoder_dictionary.pad(),
                                args.ntokens, args.dim, args.content_noise, dropout,
                                decoder_dictionary, 50, args.nconv)

    print('model loaded')

    model.eval()

    dataset, vocab = get_dataset(args.nsamples, args.data_dir, vocab)

    evaluator = Evaluator()
    fasttext_classfier = fasttext.FastText.load_model('/cs/labs/dshahaf/omribloch/data/text_lord/restorant/fasttext_model.bin')

    orig_ppl = []
    orig_bleu = []

    new_ppl = []
    new_bleu = []

    correct_counter = 0

    for i in tqdm(range(args.samples_to_eval), disable=True):
        sid = dataset[i].id
        stars = dataset[i].stars
        # stars = 1
        review_sentence = ' '.join(dataset[i].review)

        ppl, bleu, classified, soriginal, sgenerated, original_ppl = evaluator.eval(model, vocab, review_sentence, stars, sid, gready=args.gready)
        orig_ppl.append(ppl)
        orig_bleu.append(bleu)

        ppl, bleu, classified, soriginal, sgenerated_new, original_ppl = evaluator.eval(model, vocab, review_sentence, 1-stars, sid, gready=args.gready)
        new_ppl.append(ppl)
        new_bleu.append(bleu)

        predicted_label = fasttext_classfier.predict(sgenerated)[0][0]
        if labels_dictionary[predicted_label] == 1-stars:
            correct_counter += 1

        print('original - {}'.format(soriginal))
        print('reconstruct - {}'.format(sgenerated))
        print('oposite-sentiment - {}'.format(sgenerated_new))


    print(f'orig ppl: {np.average(orig_ppl)}')
    print(f'new ppl: {np.average(new_ppl)}')

    print(f'orig bleu: {np.average(orig_bleu)}')
    print(f'new bleu: {np.average(new_bleu)}')

    print(f'classifier accuracy: {correct_counter / args.samples_to_eval}')


print('haha')

if __name__ == '__main__':
    print('hoho')
    main()

