from torchtext import data
from tqdm import tqdm
import pandas
import random

START = '<s>'
END = '</s>'


def lines_generator():
    with open('/cs/labs/dshahaf/omribloch/data/text_lord/restorant/sentiment.train.0') as file:
        all_negative = file.readlines()
    with open('/cs/labs/dshahaf/omribloch/data/text_lord/restorant/sentiment.train.1') as file:
        all_positive = file.readlines()

    i = 0
    for example in all_negative:
        example = START + ' ' + example + ' ' + END
        yield {'id': i, 'stars': 0, 'review': example}
        i += 1

    for example in all_positive:
        example = START + ' ' + example + ' ' + END
        yield {'id': i, 'stars': 1, 'review': example}
        i += 1


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

    # @classmethod
    # def splits(cls, text_field, label_field, train_df, val_df=None, test_df=None, **kwargs):
    #     train_data, val_data, test_data = (None, None, None)
    #
    #     if train_df is not None:
    #         train_data = cls(train_df.copy(), text_field, label_field, **kwargs)
    #     if val_df is not None:
    #         val_data = cls(val_df.copy(), text_field, label_field, **kwargs)
    #     if test_df is not None:
    #         test_data = cls(test_df.copy(), text_field, label_field, True, **kwargs)
    #
    #     return tuple(d for d in (train_data, val_data, test_data) if d is not None)
        
# train_ds, val_ds, test_ds = DataFrameDataset.splits(
#   text_field=TEXT_FIELD, label_field=LABEL_FIELD, train_df=train_df, val_df=val_df, test_df=test_df)

# def get_dataset(vocab_size=10000, batch_size=32)
