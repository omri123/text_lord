#!/bin/csh

#SBATCH --job-name=
#SBATCH --cpus-per-task=4
#SBATCH --output=/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/run_try.out
#SBATCH --mem=16G
#SBATCH --gres=gpu:m60:1
#SBATCH --time=0:30:0

source /cs/labs/dshahaf/omribloch/env/fairseq/bin/activate.csh

python3 /cs/labs/dshahaf/omribloch/projects/text_lord/main.py --batch_size 64 --dim 64 --epochs 5 --content_lr 0.01
