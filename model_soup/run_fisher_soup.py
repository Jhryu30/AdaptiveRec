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

EXP_PATH = 'log/FINETUNE/BasicRec/ml-1m/bs256-lmd0.1-sem0.1-us_x-May-27-2023_18-07-14-lr0.001-l20-tau1-dot-DPh0.5-DPa0.5/AdaptiveRec'
MODEL_RECIPE = os.listdir(EXP_PATH)



def run_fisher(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    print('####### FIHSER MERGIGNG ######'*5)
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    config['eval_batch_size'] = 1 # set 1 to compute fisher matrix as exact as possible
    config['epochs']=10
    

        
    wandb.init(project='ModelSoup', 
            name='fisher',
            config=config, reinit=True)#, mode='disabled')
    # init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    import os
    log_dir = os.path.dirname(logger.handlers[0].baseFilename)
    config['log_dir'] = log_dir

    wandb.config.update({'log_dir':log_dir})
    
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)
    
    trainer = get_trainer(config['MODEL_TYPE'], 'basic')
    num_model = len(MODEL_RECIPE); nth_model = 0
    for recipe in MODEL_RECIPE:
        nth_model += 1
        print(f'Load {nth_model}/{num_model}:', recipe)
        recipe_path = os.path.join(EXP_PATH, recipe)
        # # trainer loading and initialization
        recipe_trainer = trainer(config, model)
        
        try:
            recipe_trainer.resume_checkpoint(resume_file=recipe_path)
        except:
            padded_ = torch.load(recipe_path, map_location=recipe_trainer.device)
            layer_ = 'item_embedding.weight'
            layer_param = padded_['state_dict'][layer_]
            padded_['state_dict'][layer_] =  layer_param[:-1]
            recipe_trainer.model.load_state_dict(padded_['state_dict'])
        
        
        recipe_model_weight = recipe_trainer.model.state_dict()
        recipe_fisher_weight = recipe_trainer.compute_fisher(eval_data=valid_data)
        
        layer_names = list(recipe_model_weight.keys())
        
        if recipe==MODEL_RECIPE[0]:
            fisher_soup ={layer : torch.mul(v,recipe_fisher_weight[layer]) for layer, v in recipe_model_weight.items()}
            devided_by = {layer : recipe_fisher_weight[layer] for layer in layer_names}
        else:
            fisher_soup = {layer : torch.mul(v,recipe_fisher_weight[layer]) + fisher_soup[layer] for layer, v in recipe_model_weight.items()}
            devided_by = {layer : recipe_fisher_weight[layer] + devided_by[layer] for layer in layer_names}
        
    fisher_soup = {k: fisher_soup[k]/(devided_by[k]) for k in list(recipe_model_weight.keys())}
    torch.save(fisher_soup, os.path.join(config['log_dir'],'fisher_model.pt'))
    
    # fisher_trainer = trainer(config, model)
    # fisher_trainer.model.load_state_dict(fisher_soup)
    
    # # model fiser finetune
    # best_valid_score, best_valid_result = fisher_trainer.fit(
    #     train_data, valid_data, saved=saved, show_progress=config['show_progress']
    # )
    # fisher_soup = fisher_trainer.model.state_dict()
    # torch.save(fisher_soup, os.path.join(config['log_dir'],'fisher_finetuned_model.pt'))
    
    # # model evaluation
    # print(config['eval_setting'],'-----------')
    # test_result = fisher_trainer.evaluate(eval_data=test_data, load_best_model=True, show_progress=config['show_progress'])
    # logger.info(set_color('test result', 'yellow') + f': {test_result}')
    # wandb.log({'test_result' : test_result})
    
   # FINE-TUNE FISHER (need to update configuration such as train/validation batch size)
    config['train_batch_size']=256; config['eval_batch_size']=256
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    model = get_model(config['model'])(config, train_data).to(config['device'])
    fisher_trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    fisher_trainer.model.load_state_dict(fisher_soup)
    test_result = fisher_trainer.evaluate(eval_data=test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    
    os.makedirs(os.path.join(config['log_dir'], config['model']), exist_ok=True)
    # model fiser finetune
    best_valid_score, best_valid_result = fisher_trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )
    fisher_soup = fisher_trainer.model.state_dict()
    # torch.save(fisher_soup, os.path.join(config['log_dir'],'fisher_finetuned_model.pt'))
    
    # model evaluation
    print(config['eval_setting'],'----------------------------------------------------------------------------------------')
    test_result = fisher_trainer.evaluate(eval_data=test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    wandb.log({'fisher_test_full_result' : test_result})
    
    
    
    # random setting (uni100)
    if config['eval_uniform_setting'] == True:
        config['eval_setting'] = 'TO_LS,uni100' 
        config['eval_batch_size'] = 256
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        model = get_model(config['model'])(config, train_data).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        trainer.model.load_state_dict(fisher_soup)
        
        print(config['eval_setting'],'-----------')
        test_random_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
        
        logger.info(set_color('test random result', 'yellow') + f': {test_random_result}')
        wandb.log({'fisher_test_random_result':test_random_result})



    # popular setting (pop100)
    if config['eval_popular_setting'] == True:
        config['eval_setting'] = 'TO_LS,pop100' 
        config['eval_batch_size'] = 256
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        model = get_model(config['model'])(config, train_data).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        trainer.model.load_state_dict(fisher_soup)
        
        print(config['eval_setting'],'-----------')
        test_popular_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
        
        logger.info(set_color('test popular result', 'yellow') + f': {test_popular_result}')
        wandb.log({'fisher_test_popular_result':test_popular_result})
        

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
