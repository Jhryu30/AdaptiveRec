import argparse

from modelsoup.run_finetune import run_finetune


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BasicRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='name of datasets')
    parser.add_argument('--log_root', type=str, default='./log/finetune/', help='name of finetune directory')
    parser.add_argument('--epochs', type=int, default=50, help='epochs for each finetuning models')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')

    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_finetune(model=args.model, dataset=args.dataset, config_file_list=config_file_list)

    
