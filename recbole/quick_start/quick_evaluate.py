# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import logging
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color

import wandb



def run_recbole_eval(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, log_dir=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save thelog_dir' model
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    # init_seed(config['seed'], config['reproducibility'])
    
    # # logger initialization
    # init_logger(config)
    # logger = getLogger()

    import os
    # log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    config['log_dir'] = log_dir
    
    # logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    # logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    # logger.info(model)
    
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    # best_valid_score, best_valid_result = trainer.fit(
    #     train_data, valid_data, saved=saved, show_progress=config['show_progress']
    # )
    trainer.resume_checkpoint(resume_file=os.path.join(config['log_dir'], 'model.pth'))

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    # logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    # logger.info(set_color('test result', 'yellow') + f': {test_result}')
    
    eval_setting = config['eval_setting'].split(',')[1]
    wandb.log({f'test_result_{eval_setting}':test_result})

    return {
        # 'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        # 'best_valid_result': best_valid_result,
        'test_result': test_result
    }



def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict): parameters dictionary used to modify experiment parameters
        config_file_list (list): config files used to modify experiment parameters
        saved (bool): whether to save the model
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
