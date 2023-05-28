import argparse

from recbole.quick_start import quick_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BasicRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')
    parser.add_argument('--target_path', type=str, 
                        default='log/FINETUNE/BasicRec/ml-1m/bs256-lmd0.1-sem0.1-us_x-May-27-2023_18-07-14-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/AdaptiveRec_model.pth',
                        help='target model directory')

    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    quick_test.run_test(model=args.model, dataset=args.dataset, config_file_list=config_file_list, target_model=args.target_path)

    