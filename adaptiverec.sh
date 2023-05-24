python run_seq.py --dataset='ml-1m' --train_batch_size=256 --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='us_x' --sim='dot' --tau=1

python run_seq.py --dataset='ml-1m' --train_batch_size=256 --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='un' --sim='dot' --tau=1

python run_seq.py --dataset='ml-1m' --train_batch_size=256 --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='su' --sim='dot' --tau=1

python run_seq.py --dataset='ml-1m' --train_batch_size=256 --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='us_x' --sim='cos' --tau=1

python run_seq.py --dataset='ml-1m' --train_batch_size=256 --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='un' --sim='cos' --tau=1

python run_seq.py --dataset='ml-1m' --train_batch_size=256 --lmd=0.1 --lmd_sem=0.1 --model='AdaptiveRec' --contrast='su' --sim='cos' --tau=1

