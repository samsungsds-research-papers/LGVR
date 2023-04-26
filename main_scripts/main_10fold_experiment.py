import os
import sys
import torch
import numpy as np

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import doc_utils
from utils.utils import get_args

from torchph import pershom
ph = pershom.pershom_backend.__C.VertFiltCompCuda__vert_filt_persistence_batch

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        # args.config = 'configs/10fold_config.json'
        config = process_config(args.config, args.dataset_name)

    except Exception as e:
        print("missing or invalid arguments {}".format(e))
        exit(0)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # TODO uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    torch.manual_seed(100)
    np.random.seed(100)
    # torch.backends.cudnn.deterministic = True  # can impact performance
    # torch.backends.cudnn.benchmark = False  # can impact performance

    print("lr = {0}".format(config.hyperparams.learning_rate))
    print("decay = {0}".format(config.hyperparams.decay_rate))
    print(config.architecture)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    doc_utils.doc_used_config(config)
    for exp in range(1, config.num_exp+1):
        for fold in range(1, 11):
            print("Experiment num = {0}\nFold num = {1}".format(exp, fold))
            # create your data generator
            config.num_fold = fold
            data = DataGenerator(config)
            # create an instance of the model you want
            model_wrapper = ModelWrapper(config, data, ph)
            # create trainer and pass all the previous components to it
            trainer = Trainer(model_wrapper, data, config)
            # here you train your model
            trainer.train()
    # doc_utils.summary_10fold_results(config.summary_dir)

if __name__ == '__main__':
    main()
