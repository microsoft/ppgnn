#!/bin/bash
python train_model.py --dataset cora_full_pca --net PPGNN --split 4 --hidden 64 --Init PPR --alpha 0.5 --lr 0.001 --weight_decay 0.001 --dropout 0.5 --total_buckets 4 --alphas "0.891402077, 0.497676851, 0.979201634, 0.801711466, 0.166593359" --evd_dims 1024 --beta 0.524618276 --K 8

