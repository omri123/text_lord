import pandas

df = pandas.read_csv('/home/omribloch/data/yelp/csv/test_tmp.csv', sep=',',  names=['stars', 'review'])
print(df.shape)
df = df.sample(frac=1)
df.to_csv('/home/omribloch/data/yelp/csv/test_tmp_shuffled.csv')