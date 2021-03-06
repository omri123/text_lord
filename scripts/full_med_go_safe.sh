#!/bin/csh

#SBATCH --job-name='fmgs'
#SBATCH --cpus-per-task=8
#SBATCH --output=/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/full_med_go_safe.out
#SBATCH --mem=20G
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=1-0

source /cs/labs/dshahaf/omribloch/env/fairseq/bin/activate.csh

python /cs/labs/dshahaf/omribloch/projects/text_lord/main.py --overwrite --note full_med_go_safe --batch_size 256 --epochs 10 --content_wdecay 0.001 --dim 256 --content_noise 0.5 --nsamples 50000 --device cuda:0 --nconv 10 --it 30 -f