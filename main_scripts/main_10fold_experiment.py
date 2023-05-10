import os
import sys
import torch
import numpy as np
from datetime import datetime

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
from utils.config import process_config, create_save_path
from utils.dirs import create_dirs
from utils import doc_utils
from utils.utils import get_args
from torchph import pershom

pvr = pershom.pershom_backend.__C.VRCompCuda__vr_persistence

def main():
    try:
        args = get_args()
        config = process_config(args.config, args.dataset_name)
    except Exception as e:
        print("missing or invalid arguments {}".format(e))
        exit(0)

    torch.manual_seed(100)
    np.random.seed(100)

    lr_list = config.hyperparams.learning_rate
    for lr in lr_list:
        config = process_config(args.config, args.dataset_name)
        config.hyperparams.learning_rate = lr
        config = create_save_path(config)
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
                model_wrapper = ModelWrapper(config, data, pvr=pvr)
                # create trainer and pass all the previous components to it
                trainer = Trainer(model_wrapper, data, config)
                # here you train your model
                trainer.train()

        doc_utils.summary_10fold_results(config.summary_dir)


if __name__ == '__main__':
    start = datetime.now()
    main()
    print('Runtime: {}'.format(datetime.now() - start))
