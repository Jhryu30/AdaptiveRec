import os

import argparse

from modelsoup.run_fisher_soup import run_fisher_model_soup
from modelsoup.run_uniform_soup import run_uniform_model_soup



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BasicRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='name of datasets')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of finetuning epochs after merging')
    
    parser.add_argument('--do_fisher_merging', type=bool, default=False)
    parser.add_argument('--do_uniform_merging', type=bool, default=False)
    
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    
    EXP_PATH = 'log/AdaptiveRec/ml-1m'
    MODEL_RECIPE = os.listdir(EXP_PATH)
    
    if args.do_fisher_merging:
        run_fisher_model_soup(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
        
    if args.do_uniform_merging:
        run_uniform_model_soup(model=args.model, dataset=args.dataset, config_file_list=config_file_list)