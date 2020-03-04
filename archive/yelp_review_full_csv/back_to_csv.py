import argparse

# parser = argparse.ArgumentParser(description='Concatenate all examples from yelp dataset for BPE building.')
# parser.add_argument('--textfile', type=str, help='encoded, concatenated file')
# parser.add_argument('--starsfile', type=str, help='each line is number of stars')
# parser.add_argument('--outfile', type=str, help='the csv output file path')
# args = parser.parse_args()

stars = '/cs/labs/dshahaf/omribloch/data/text_lord/concat/train.txt.stars'
texts = [
    '/cs/labs/dshahaf/omribloch/data/text_lord/concat/train_encoded_2k.txt',
    '/cs/labs/dshahaf/omribloch/data/text_lord/concat/train_encoded_10k.txt',
    '/cs/labs/dshahaf/omribloch/data/text_lord/concat/train_encoded_32k.txt'
]
outfiles = [
    '/cs/labs/dshahaf/omribloch/data/text_lord/final/train_2k.csv',
    '/cs/labs/dshahaf/omribloch/data/text_lord/final/train_10k.csv',
    '/cs/labs/dshahaf/omribloch/data/text_lord/final/train_32k.csv'
]

def work(stars, text, out):
    with open(text, 'r') as textfile, open(stars, 'r') as starsfile, open(out, 'w') as outfile:
        for i in range(650000):
            line = textfile.readline().strip('\n')
            star = starsfile.readline().strip('\n')
            # star = int(starsfile.readline().strip('\n'))
            # if i % 1000 == 0:
            #     print(star)
            #     print(i)
            #     print('--')

            newstring = f'"{star}","{line}"\r'
            outfile.write(newstring)

for text, out in zip(texts, outfiles):
    print(out)
    work(stars, text, out)

