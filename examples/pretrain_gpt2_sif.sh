singularity exec --nv --bind $PWD/examples:/scripts build/fmtrainer.sif python3 /scripts/pretrain_gpt2.py