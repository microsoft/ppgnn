#!/bin/bash
python train_model.py \
    --dataset flickr \
    --net PPGNN \
    --split 0 \
    --hidden 64 \
    --Init PPR \
    --alpha 0.5 \
    --lr 0.003 \
    --weight_decay 0.0005 \
    --dropout 0.2 \
    --total_buckets 5 \
    --alphas "0.211859634, 0.747164365, 0.261621357, 0.347239897, 0.262261408, 0.211787549" \
    --evd_dims 256 \
    --beta 0.460230698 \
    --K 3