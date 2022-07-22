import sys
sys.path.append('/home/name/k_space_da_main')
import os
from configs.assets.kswap_3d import *
from kswap.easy_swapping import DA_experiment
from tqdm import tqdm
import json

optimal_beta_per_pair = {'2d': {'sm3_ge15': [0.025, 0.01],
  'ge3_ph3': [0.025, 0.01],
  'ph3_ge15': [0.01, 0.005],
  'sm15_ph3': [0.01, 0.005],
  'ph3_sm15': [0.01, 0.005],
  'ge15_ph15': [0.01, 0.005]},
 '25d': {'sm3_ge15': [0.015, 0.01],
  'ge3_ph3': [0.025, 0.01],
  'ph3_ge15': [0.01, 0.005],
  'sm15_ph3': [0.01, 0.005],
  'ph3_sm15': [0.01, 0.005],
  'ge15_ph15': [0.01, 0.005]},
 '3d': {'sm3_ge15': [0.025, 0.015],
  'ge3_ph3': [0.025, 0.015],
  'ph3_ge15': [0.015, 0.005],
  'sm15_ph3': [0.015, 0.005],
  'ph3_sm15': [0.015, 0.005],
  'ge15_ph15': [0.015, 0.005]},
 'mst': {'sm3_ge15': [0.015, 0.005],
  'ge3_ph3': [0.015, 0.005],
  'ph3_ge15': [0.005],
  'sm15_ph3': [0.005],
  'ph3_sm15': [0.005],
  'ge15_ph15': [0.005]},
 'none': {'sm3_ge15': [0.015, 0.005],
  'ge3_ph3': [0.005],
  'ph3_ge15': [0.005],
  'sm15_ph3': [0.005],
  'ph3_sm15': [0.005],
  'ge15_ph15': [0.005]}}

n_scans_dict = {'2d': 7, '25d': 7, '3d': 7, 'mst': 7, 'none': 1}
mode = 'test'

closest_scans_mode = '2d' # 2d OR 3d or 25d or None
n_scans = n_scans_dict[closest_scans_mode]

for pair in tqdm(pairs):

    source = pair[0]
    target = pair[1]

    exp_name = id2dom[source] + '_' + id2dom[target]

    for beta in optimal_beta_per_pair[closest_scans_mode][exp_name]:

        obj = DA_experiment(sourceID=source, targetID=target, dataset=dataset, predict=predict,
                            sdice_metric=sdice_metric, mode=mode, closest_scans_mode=closest_scans_mode,
                            n_scans=n_scans)

        basedir = os.path.join('/mnt/nfs_storage/name/exps/naive_swap", mode' +
                               '_n_' + str(n_scans) + '_' + closest_scans_mode, exp_name)

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        path_final = os.path.join(basedir, str(beta) + '.json')
        if os.path.exists(path_final):
            pass
        else:
            res = obj.exp(obj.swap_low_freq_circle, beta=beta, with_phase=False)
            with open(path_final, 'w') as f:
                json.dump(res, f)

        print (str(source) + '->' + str(target) + '; beta = ' + str(beta) + ' done', flush=True)
