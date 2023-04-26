import os
import sys
import argparse

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

import pandas as pd


def summary_10fold_results(filepath):
    df = pd.read_csv(os.path.join(filepath, 'per_epoch_stats.csv'))
    try:
        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        df.drop(['timestamp'], axis=1, inplace=True)
    except:
        print("No suitable columns")
    df.to_csv(os.path.join(filepath, 'per_epoch_stats.csv'), index=False)

    df = df.drop(['train_loss', 'train_accuracy', 'experiment_name'], axis=1)  # drop irrelevant columns
    # df = df.drop(['train_loss', 'train_accuracy', 'timestamp', 'experiment_name'], axis=1)  # drop irrelevant columns
    df_group_std = df.groupby('epoch').std()
    df_group = df.groupby('epoch').mean()
    df_group['std'] = df_group_std.val_accuracy
    best_epoch = df_group['val_accuracy'].idxmax()

    best_row = df_group.loc[best_epoch]
    print("Results")
    print("Best epoch - {0}".format(best_epoch))
    print("Mean Accuracy = {0}".format(best_row['val_accuracy']))
    print("Mean std = {0}".format(best_row['std']))

    # Document the validation results of the best epoch, per experiment
    df2 = df[df.epoch == best_epoch].copy()
    df2['fold'] = pd.Series(range(10), index=df2.index) + 1
    df2 = df2.append(pd.Series([best_epoch, best_row['val_loss'], best_row['val_accuracy'], 'mean'], index=df2.columns),
                     ignore_index=True)
    fullpath = os.path.join(filepath, 'exp_summary.csv')
    df2.to_csv(fullpath, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='MUTAG', type=str, help="Available options: ['MUTAG', 'PTC', 'PROTEINS', 'NCI1', 'NCI109', 'IMDBMULTI', 'IMDBBINARY', 'ENZYMES']")
    parser.add_argument("--default_path", default=None, type=str, help='file path for experimental results')

    args = parser.parse_args()
    filepath = args.default_path + args.data_name

    summary_10fold_results(filepath)

if __name__ == '__main__':
    main()
