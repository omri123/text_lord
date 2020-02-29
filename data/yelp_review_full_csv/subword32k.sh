#/bin/bash

conda activate Zion
subword-nmt learn-bpe -s 32000 < /home/omribloch/data/yelp/vocab/concatenated.txt > /home/omribloch/data/yelp/vocab/32k.bpe
subword-nmt apply-bpe -c /home/omribloch/data/yelp/vocab/32k.bpe < /home/omribloch/data/yelp/vocab/concatenated.txt > /home/omribloch/data/yelp/vocab/concatenated_encoded.txt