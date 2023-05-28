import argparse

from model_soup.run_fisher_soup import run_fisher
from model_soup.run_uniform_soup import run_uniform




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BasicRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    
    run_uniform(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    run_fisher(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    