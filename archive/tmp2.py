import pandas

df = pandas.read_csv('/home/omribloch/data/yelp/csv/test_tmp_short.csv', names=['stars', 'review'])
print(df)