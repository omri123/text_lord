srun -c8 --mem=24G --gres=gpu:rtx2080:1 --pty $SHELL
cd /cs/labs/dshahaf/omribloch ; source ./env/fairseq/bin/activate.csh ;
setenv XDG_RUNTIME_DIR /cs/usr/omribloch/tmp
jupyter-notebook --no-browser --ip=0.0.0.0
