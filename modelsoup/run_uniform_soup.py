# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import os
import sys
sys.path.append('../')

import torch

import logging
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color

import wandb


def run_uniform_model_soup(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    config['log_root'] = './log/modelsoup/uniform/'
        
    wandb.init(project='ModelSoup', 
            name=config.model+config['contrast']+config['eval_setting'].split(',')[1],
            config=config, mode='disabled')
    # init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    import os
    log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    config['log_dir'] = log_dir

    wandb.config.update({'log_dir':log_dir})
    
    logger.info(config)

    num_model = len(MODEL_RECIPE)
    
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)
    
    trainer = get_trainer(config['MODEL_TYPE'], 'basic')
    for recipe in MODEL_RECIPE:
        recipe_path = os.path.join(EXP_PATH, recipe)
        # # trainer loading and initialization
        recipe_trainer = trainer(config, model)
        recipe_trainer.resume_checkpoint(resume_file=os.path.join(recipe_path, 'model.pth'))
        recipe_model_weight = recipe_trainer.model.state_dict()
        
        
        if recipe==MODEL_RECIPE[0]:
            uniform_soup ={k : v for k, v in recipe_model_weight.items()}
        else:
            uniform_soup = {k : v+uniform_soup[k] for k, v in recipe_model_weight.items()}
            
    
    
    uniform_trainer = trainer(config, model)
    
    uniform_trainer.model.load_state_dict(uniform_soup)
    uniform_trainer.save_model_file =os.path.join(config['log_dir'],'uniform_model.pth')
    uniform_trainer._save_checkpoint(epoch=0)
    
    
    
    # model fiser finetune
    uniform_trainer.save_model_file =os.path.join(config['log_dir'],'uniform_finetune_model.pth')
    best_valid_score, best_valid_result = uniform_trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    uniform_soup = uniform_trainer.model.state_dict()
    
    # model evaluation
    print(config['eval_setting'],'-----------')
    test_result = uniform_trainer.evaluate(eval_data=test_data, load_best_model=True, show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    wandb.log({'test_result' : test_result})
    
    
    
    # random setting (uni100)
    if config['eval_uniform_setting'] == True:
        config['eval_setting'] = 'TO_LS,uni100' 
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        model = get_model(config['model'])(config, train_data).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        trainer.model.load_state_dict(uniform_soup)
        
        print(config['eval_setting'],'-----------')
        test_random_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
        
        logger.info(set_color('test random result', 'yellow') + f': {test_random_result}')
        wandb.log({'test_random_result':test_random_result})



    # popular setting (pop100)
    if config['eval_popular_setting'] == True:
        config['eval_setting'] = 'TO_LS,pop100' 
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        model = get_model(config['model'])(config, train_data).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        trainer.model.load_state_dict(uniform_soup)
        
        print(config['eval_setting'],'-----------')
        test_popular_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
        
        logger.info(set_color('test popular result', 'yellow') + f': {test_popular_result}')
        wandb.log({'test_popular_result':test_popular_result})
        

    return {
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
