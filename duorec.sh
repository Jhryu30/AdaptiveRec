#!/bin/sh

batch_size=1024
dataset='yelp2022'
CONTRAST="su un us_x"


for contrast in $CONTRAST
do
    CUDA_VISIBLE_DEVICES=1 python run_seq.py --dataset=$dataset --train_batch_size=$batch_size --model='DuoRec' --contrast=$contrast --sim='cos' --gpu_id=0 \
                                                --lmd=0.1 --lmd_sem=0.1 --tau=1 \
                                                --name='DuoRec'\_$contrast
done



# CUDA_VISIBLE_DEVICES=0 python run_seq.py --dataset=$dataset --train_batch_size=$batch_size --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='su' --sim='cos' --gpu_id=0  --tau=1

# CUDA_VISIBLE_DEVICES=0 python run_seq.py --dataset=$dataset --train_batch_size=$batch_size --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='un' --sim='cos' --gpu_id=0  --tau=1

# CUDA_VISIBLE_DEVICES=0 python run_seq.py --dataset=$dataset --train_batch_size=$batch_size --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='us_x' --sim='cos' --gpu_id=0  --tau=1

