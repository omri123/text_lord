#!/bin/csh

#SBATCH --job-name=med-0.05
#SBATCH --cpus-per-task=4
#SBATCH --output=/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/run_try.out
#SBATCH --mem=16G
#SBATCH --gres=gpu:m60:1
#SBATCH --time=8:00:0

source /cs/labs/dshahaf/omribloch/env/fairseq/bin/activate.csh

python3 /cs/labs/dshahaf/omribloch/projects/text_lord/main.py --batch_size 64 --dim 128 --epochs 1000 --content_lr 0.05
