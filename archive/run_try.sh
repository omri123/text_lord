#!/bin/csh

#SBATCH --cpus-per-task=8
#SBATCH --output=/cs/labs/dshahaf/omribloch/data/text_lord/restorant/train/run_try.out
#SBATCH --mem=16G
#SBATCH --gres=gpu:m60:1
#SBATCH --time=0:1:0

source /cs/labs/dshahaf/omribloch/env/fairseq/bin/activate.csh

python3 /cs/labs/dshahaf/omribloch/projects/text_lord/main.py --batch_size 64 --dim 64 --epochs 5
