#!/bin/csh

#SBATCH --job-name='fls2'
#SBATCH --cpus-per-task=8
#SBATCH --output=/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/full_large_shallow_2.out
#SBATCH --mem=20G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=1-0

source /cs/labs/dshahaf/omribloch/env/fairseq/bin/activate.csh
python /cs/labs/dshahaf/omribloch/projects/text_lord/main_partitioned.py --overwrite --note full_large_shallow_2 --batch_size 512 --epochs 10 --it 30 --content_wdecay 0.001 --dim 256 --content_noise 0.5 --nsamples 500000 --device cuda:0 --nconv 10 -f --ntokens 5 --shuffle --dropout 0.1