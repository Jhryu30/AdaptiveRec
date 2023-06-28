#!/bin/sh

batch_size=1024
dataset='amazon_beauty'

CUDA_VISIBLE_DEVICES=3 python run_seq.py --dataset=$dataset --train_batch_size=$batch_size --lmd=0.1 --lmd_sem=0.1 --model='CL4SRec' --contrast='su' --sim='cos' --gpu_id=0