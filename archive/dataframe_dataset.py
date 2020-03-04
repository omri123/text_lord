from torchtext import data
from tqdm import tqdm
import pandas

START = 'START'
END = ' END'

def yelp_generator(csv_path):
    reader = pandas.read_csv(csv_path, sep=',', names=['stars', 'review'], chunksize=10 ** 4)
    example_id = 0
    print('reading')
    for df in reader:
        for index, row in df.iterrows():
            review = START + ' ' + row['review'] + ' ' + END
            stars = row['stars']
            yield {"review": review, "stars": stars, "id": example_id}
            example_id += 1


class DataFrameDataset(data.Dataset):

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
