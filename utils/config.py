import json
from easydict import EasyDict
import os
import datetime

NUM_LABELS = {'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':7, 'NCI1':37, 'NCI109':38, 'PROTEINS':3, 'PTC':22, 'ENZYMES':3, 'QM9': 18}
NUM_CLASSES = {'IMDBBINARY':2, 'IMDBMULTI':3, 'MUTAG':2, 'NCI1':2, 'NCI109':2, 'PROTEINS':2, 'PTC':2, 'ENZYMES':6, 'QM9': 12}
LEARNING_RATES = {'IMDBBINARY': 0.00005, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC': 0.0001, 'ENZYMES':0.0001}
DECAY_RATES = {'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 1.0, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC': 1.0, 'ENZYMES':0.5}
CHOSEN_EPOCH = {'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC': 400, 'ENZYMES':400}
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
    if config.dataset_name == 'QM9' and config.target_param is not False:
        config.num_classes = 1
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    config.summary_dir = os.path.join("../experiments", config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.parent_dir, "checkpoint/")
    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        config.hyperparams.learning_rate = LEARNING_RATES[config.dataset_name]
        config.hyperparams.decay_rate = DECAY_RATES[config.dataset_name]
    config.n_gpus = len(config.gpu.split(','))
    # config.n_gpus = 0
    if config.n_gpus > 0:
        config.gpus_list = ",".join(['{}'.format(i) for i in range(config.n_gpus)])
        config.devices = ['/gpu:{}'.format(i) for i in range(config.n_gpus)]
    else:
        config.devices = 'cpu'
    config.distributed_fold = None  # specific for distrib 10fold - override to use as a flag
    return config


if __name__ == '__main__':
    config = process_config('../configs/10fold_config.json', 'NCI1')
    for k, v in config.items():
        print(k, ": ", v)
    # print(config.values())
