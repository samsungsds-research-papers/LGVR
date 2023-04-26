import os

datasets = ['MUTAG', 'PROTEINS', 'PTC', 'NCI1', 'IMDBBINARY', 'IMDBMULTI']
run_command = 'python main_scripts/main_10fold_experiment.py '
args1 = '--config=configs/10fold_config_gin_lgvr_plus.json '
args2 = '--dataset_name=%s'

def main():
    os.system('pwd')
    for i in range(len(datasets)):
        command = run_command + args1 + args2 % datasets[i]
        os.system(command)

if __name__ == '__main__':
    main()
