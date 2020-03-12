#!/bin/csh

#SBATCH --job-name='fls'
#SBATCH --cpus-per-task=8
#SBATCH --output=/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/partial_large_shallow.out
#SBATCH --mem=20G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=1-0

source /cs/labs/dshahaf/omribloch/env/fairseq/bin/activate.csh
python /cs/labs/dshahaf/omribloch/projects/text_lord/main_partitioned.py --overwrite --note partial_large_shallow_partitioned --batch_size 1024 --epochs 10 --it 50 --content_wdecay 0.001 --dim 256 --content_noise 0.5 --nsamples 50000 --device cuda:0 --nconv 10 -f --ntokens 5 --shuffle --dropout 0.1