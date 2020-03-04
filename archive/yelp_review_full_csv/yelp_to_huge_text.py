import pandas
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Concatenate all examples from yelp dataset for BPE building.')
parser.add_argument('--infile', type=str, help='Input file, csv format')
parser.add_argument('--outfile', type=str, help='output file, txt format')
args = parser.parse_args()


with open(args.outfile, 'w') as outfile, open(args.outfile + '.stars', 'w') as starsfile:

    reader = pandas.read_csv(args.infile, sep=',', names=['stars', 'review'], chunksize=10 ** 4)

    for df in tqdm(reader):
        for index, sample in df.iterrows():
            line = sample['review']
            stars = sample['stars']
            # line = line.replace('\\n', ' \\n ')
            # line = line.replace('\\"', ' \\" ')
            line += "\n"
            outfile.write(line)
            stars_string = '{}\n'.format(stars)
            starsfile.write(stars_string)
        # break

