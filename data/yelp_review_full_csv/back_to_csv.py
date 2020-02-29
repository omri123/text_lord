import argparse

parser = argparse.ArgumentParser(description='Concatenate all examples from yelp dataset for BPE building.')
parser.add_argument('--textfile', type=str, help='encoded, concatenated file')
parser.add_argument('--starsfile', type=str, help='each line is number of stars')
parser.add_argument('--outfile', type=str, help='the csv output file path')
args = parser.parse_args()

with open(args.textfile, 'r') as textfile, open(args.starsfile, 'r') as starsfile, open(args.outfile, 'w') as outfile:
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

