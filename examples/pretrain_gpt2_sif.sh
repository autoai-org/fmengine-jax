singularity exec --nv \
--bind $PWD/examples:/scripts \
--bind $PWD/.cache:/cache build/fmtrainer.sif \
python3 /scripts/pretrain_gpt2.py