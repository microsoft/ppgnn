#!/bin/bash
python train_model.py --dataset amazon_electronics_computers --net PPGNN --split 2 --hidden 32 --Init PPR --alpha 0.1 --lr 0.008 --weight_decay 0.005 --dropout 0.3 --total_buckets 2 --alphas "0.132110701, 0.62285746, 0.142224807" --evd_dims 32 --beta 0.642958427 --K 3
