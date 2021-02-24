#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import flags

import os
import sys

import config.const as const_util

import numpy as np
from scipy.stats import entropy


class Judger(object):

    def __init__(self, flags_obj, dm, topk):

        self.name = flags_obj.name + '_judger'
        self.metrics = flags_obj.metrics
        self.dm = dm
        self.topk = topk
        self.workspace = flags_obj.workspace

    def judge(self, items, test_pos, num_test_pos):

        results  = {}
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            results[metric] = sum([f(items[i], test_pos=test_pos[i], num_test_pos=num_test_pos[i].item()) if num_test_pos[i] > 0 else 0 for i in range(len(items))])

        valid_num_users = sum([1 if len(t) > 0 else 0 for t in test_pos])

        return results, valid_num_users


class Metrics(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_metrics'

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'ndcg': Metrics.ndcg
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def ndcg(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']

        index = np.arange(len(items))
        k = min(len(items), num_test_pos)
        idcg = (1/np.log(2 + np.arange(k))).sum()
        dcg = (1/np.log(2 + index[np.isin(items, test_pos)])).sum()

        return dcg/idcg
