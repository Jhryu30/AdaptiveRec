import argparse

from recbole.quick_start import run_recbole
from recbole.quick_start.quick_evaluate import run_recbole_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DuoRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    result, log_dir = run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)

    
    # random setting
    args.eval_setting ='TO_LS,uni100'
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_eval(model=args.model, dataset=args.dataset, config_file_list=config_file_list, log_dir=log_dir)
    
    # popular setting
    args.eval_setting ='TO_LS,pop100'
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_eval(model=args.model, dataset=args.dataset, config_file_list=config_file_list, log_dir=log_dir)
