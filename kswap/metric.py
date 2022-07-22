import os
from collections import defaultdict
from typing import Sequence, Callable

import numpy as np
from tqdm import tqdm

from dpipe.commands import load_from_folder
from dpipe.io import save_json
from dpipe.itertools import zip_equal


def aggregate_metric_probably_with_ids(xs, ys, ids, metric, aggregate_fn=np.mean):
    """Aggregate a `metric` computed on pairs from `xs` and `ys`"""
    try:
        return aggregate_fn([metric(x, y, i) for x, y, i in zip_equal(xs, ys, ids)])
    except TypeError:
        return aggregate_fn([metric(x, y) for x, y in zip_equal(xs, ys)])


def compute_metrics_probably_with_ids(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str],
                                      metrics: dict):
    return evaluate_with_ids(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], ids, metrics)


def evaluate_with_ids(y_true: Sequence, y_pred: Sequence, ids: Sequence[str], metrics: dict) -> dict:
    return {name: metric(y_true, y_pred, ids) for name, metric in metrics.items()}

def evaluate_individual_metrics_probably_with_ids(load_y_true, metrics: dict, predictions_path, results_path,
                                                  exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        target = load_y_true(identifier)

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][identifier] = metric(target, prediction, identifier)
            except TypeError:
                results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)

def evaluate_individual_metrics_probably_with_ids_no_pred(load_y, load_x, predict, metrics: dict, ids_source,
                                                          ids_target, results_path, exist_ok=False):
    if ids_source is not None:

        assert len(metrics) > 0, 'No metric provided'
        os.makedirs(results_path, exist_ok=exist_ok)

        results = defaultdict(dict)
        for _id_s in ids_source:
            for _id_t in tqdm(ids_target):
                scan_s, scan_t = load_x(_id_s), load_x(_id_t)
                target = load_y(_id_t)
                prediction = predict(scan_s, scan_t)

                for metric_name, metric in metrics.items():
                    try:
                        results[metric_name][_id_s + '_' + _id_t] = metric(target, prediction, _id_t)
                    except TypeError:
                        results[metric_name][_id_s + '_' + _id_t] = metric(target, prediction)

        for metric_name, result in results.items():
            save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)

    else:
        raise NotImplementedError
