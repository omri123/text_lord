from torchtext import data
from tqdm import tqdm
import os

START = '<s>'
END = '</s>'


def lines_generator(path):
    with open(os.path.join(path, 'sentiment.train.0')) as file:
        all_negative = file.readlines()
    with open(os.path.join(path, 'sentiment.train.1')) as file:
        all_positive = file.readlines()

    for i in range(min(len(all_negative), len(all_positive))):

        example = START + ' ' + all_negative[i] + ' ' + END
        yield {'id': 2*i, 'stars': 0, 'review': example}

        example = START + ' ' + all_positive[i] + ' ' + END
        yield {'id': 2*i + 1, 'stars': 1, 'review': example}
    # i = 0
    # for example in all_negative:
    #     example = START + ' ' + example + ' ' + END
    #     yield {'id': i, 'stars': 0, 'review': example}
    #     i += 1
    #
    # for example in all_positive:
    #     example = START + ' ' + example + ' ' + END
    #     yield {'id': i, 'stars': 1, 'review': example}
    #     i += 1


class RestDataset(data.Dataset):

    def __init__(self, example_generator, id_field, stars_field, review_field, max_examples, is_test=False, **kwargs):
        fields = [('id', id_field), ('stars', stars_field), ('review', review_field)]
        examples = []
        for i, example in enumerate(tqdm(example_generator)):
            if i == max_examples:
                break
            example_id = example['id']
            stars = example['stars']
            review = example['review']
            examples.append(data.Example.fromlist([example_id, stars, review], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.review)


def get_dataset(max_examples, path, vocab=None):
    g = lines_generator(path=path)

    id_f = data.Field(sequential=False, use_vocab=False)
    stars_f = data.Field(sequential=False, use_vocab=False)
    review_f = data.Field(sequential=True, use_vocab=True)

    dataset = RestDataset(g, id_f, stars_f, review_f, max_examples)

    if vocab:
        review_f.vocab = vocab
    else:
        review_f.build_vocab(dataset)

    return dataset, review_f.vocab