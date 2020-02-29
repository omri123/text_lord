
with open('/home/omribloch/data/yelp/csv/test_tmp_short.csv', 'w') as outfile:
    with open('/home/omribloch/data/yelp/csv/test_tmp.csv', 'r') as infile:
        for i in range(9):
            l = infile.readline()
            outfile.write(l)
