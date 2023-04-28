import json
from utils.utils import EasyDict
import os
import datetime

NUM_LABELS = {'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':7, 'NCI1':37, 'NCI109':38, 'PROTEINS':3, 'PTC':22, 'QM9': 18}
NUM_CLASSES = {'IMDBBINARY':2, 'IMDBMULTI':3, 'MUTAG':2, 'NCI1':2, 'NCI109':2, 'PROTEINS':2, 'PTC':2, 'QM9': 12}
DECAY_RATES = {'GIN': {'IMDBBINARY': 0.75, 'IMDBMULTI': 0.75, 'MUTAG': 0.75, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.75, 'PTC': 0.75},
               'PPGN': {'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 1.0, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC': 1.0}}
CHOSEN_EPOCH = {'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109':250, 'PROTEINS': 100, 'PTC': 400}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config

def process_config(json_file, dataset_name):
    config = get_config_from_json(json_file)
    if dataset_name != '':
        config.dataset_name = dataset_name
    config.num_classes = NUM_CLASSES[config.dataset_name]
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME

    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        if config.gin:
            config.hyperparams.decay_rate = DECAY_RATES['GIN'][config.dataset_name]
        else:
            config.hyperparams.decay_rate = DECAY_RATES['PPGN'][config.dataset_name]

    return config

def create_save_path(config):
    config.parent_dir = 'lr_' + str(config.hyperparams.learning_rate)
    if config.model == 'gin_lgvr':
        config.summary_dir = os.path.join("experiments/GIN_LGVR/" + str(config.dataset_name), config.parent_dir, "summary/")
        config.checkpoint_dir = os.path.join("experiments/GIN_LGVR/" + str(config.dataset_name), config.parent_dir, "checkpoint/")
    elif config.model == 'gin_lgvr_plus':
        config.summary_dir = os.path.join("experiments/GIN_LGVR_plus/" + str(config.dataset_name), config.parent_dir, "summary/")
        config.checkpoint_dir = os.path.join("experiments/GIN_LGVR_plus/" + str(config.dataset_name), config.parent_dir, "checkpoint/")
    elif config.model == 'ppgn_lgvr_plus':
        config.summary_dir = os.path.join("experiments/PPGN_LGVR_plus/" + str(config.dataset_name), config.parent_dir, "summary/")
        config.checkpoint_dir = os.path.join("experiments/PPGN_LGVR_plus/" + str(config.dataset_name), config.parent_dir, "checkpoint/")
    else:
        raise Exception("config.model should be one of ['gin_lgvr', 'gin_lgvr_plus', 'ppgn_lgvr_plus']")

    return config
