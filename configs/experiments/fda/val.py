import sys
sys.path.append('/home/name/k_space_da_main')
import os
from configs.assets.kswap_3d import *
from kswap.easy_swapping import DA_experiment
from tqdm import tqdm
import json
import numpy as np

mode = 'val'

closest_scans_mode = '2d' # 2d OR 3d or 25d or None
n_scans_list = [7,]
betas = np.arange(0.005, 0.04, 0.005)

pairs = [(1, 2),]

for n_scans in n_scans_list:
    for pair in pairs:

        source = pair[0]
        target = pair[1]

        obj = DA_experiment(sourceID=source, targetID=target, dataset=dataset, predict=predict,
                            sdice_metric=sdice_metric, mode=mode, closest_scans_mode=closest_scans_mode,
                            n_scans=n_scans)

        exp_name = id2dom[source] + '_' + id2dom[target]
        basedir = os.path.join('/mnt/nfs_storage/name/exps/naive_swap', mode +
                               '_n_' + str(n_scans) + '_' + closest_scans_mode, exp_name)

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        for beta in tqdm(betas):
            path_final = os.path.join(basedir, str(beta) + '.json')
            if os.path.exists(path_final):
                pass
            else:
                res = obj.exp(obj.swap_low_freq_circle, beta=beta, with_phase=False)
                with open(path_final, 'w') as f:
                    json.dump(res, f)

            print (str(source) + '->' + str(target) + '; beta = ' + str(beta) +
                   'n = ' + str(n_scans) + '; done', flush=True)
