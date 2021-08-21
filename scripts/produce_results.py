import os
import sys
BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(BASEDIR)
split_type = sys.argv[1]

import torch
torch.set_num_threads(1)
target_names = os.listdir(os.path.join(BASEDIR, "data", "dataset"))
target_names = [n.replace(".csv", "") for n in target_names]
target_names.sort()

def run_func(target_name):
  os.system(f"python {BASEDIR}/train.py \
            --data_path {BASEDIR}/data/dataset/{target_name}.csv \
            --dataset_type regression \
            --save_dir {BASEDIR}/results/{split_type}/{target_name} \
            --save_smiles_splits \
            --ensemble_size 10 \
            --earlystop 50 \
            --batch_size 50 \
            --split_type index_predetermined \
            --crossval_index_file {BASEDIR}/data/{split_type}/{target_name}_{split_type}.pkl \
            --gpu 0 \
            --metric rmse  \
            --extra_metrics mae r2")

from multiprocessing import Pool
pool = Pool(1)
for target_name in target_names:
    pool.apply_async(func=run_func, args=(target_name,))
pool.close()
pool.join()

