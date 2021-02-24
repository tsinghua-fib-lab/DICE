#!/usr/local/anaconda3/envs/torch-1.1-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import os
import datetime
import setproctitle

from absl import logging
from absl import flags
from visdom import Visdom

from deprecated import deprecated
from tqdm import tqdm

import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

import numbers

import torch

import config.const as const_util
import trainer
import recommender
import candidate_generator as cg

import data_utils.loader as LOADER
import data_utils.transformer as TRANSFORMER
import data_utils.sampler as SAMPLER


class ContextManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_cm'
        self.exp_name = flags_obj.name
        self.output = flags_obj.output
        self.set_load_path(flags_obj)

    @staticmethod
    def set_load_path(flags_obj):

        if flags_obj.dataset == 'ml10m':
            flags_obj.load_path = const_util.ml10m
        elif flags_obj.dataset == 'nf':
            flags_obj.load_path = const_util.nf

    def set_default_ui(self):

        self.set_workspace()
        self.set_process_name()
        self.set_logging()

    def set_workspace(self):

        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dir_name = self.exp_name + '_' + date_time
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        self.workspace = os.path.join(self.output, dir_name)
        os.mkdir(self.workspace)

    def set_process_name(self):

        setproctitle.setproctitle(self.exp_name + '@zhengyu')

    def set_logging(self):

        self.log_path = os.path.join(self.workspace, 'log')
        if not os.path.exists(self.log_path):

            os.mkdir(self.log_path)

        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.exp_name + '.log', self.log_path)

    def set_test_logging(self):

        self.log_path = os.path.join(self.workspace, 'test_log')
        if not os.path.exists(self.log_path):

            os.mkdir(self.log_path)

        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.exp_name + '.log', self.log_path)

    def logging_flags(self, flags_obj):

        logging.info('FLAGS:')
        for flag, value in flags_obj.flag_values_dict().items():
            logging.info('{}: {}'.format(flag, value))

    @staticmethod
    def set_trainer(flags_obj, cm, vm, dm):

        if 'DICE' not in flags_obj.model:
            return trainer.PairTrainer(flags_obj, cm, vm, dm)
        else:
            return trainer.DICETrainer(flags_obj, cm, vm, dm)

    @staticmethod
    def set_recommender(flags_obj, workspace, dm):

        if flags_obj.model == 'MF':
            return recommender.MFRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'DICE':
            return recommender.DICERecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'LGN':
            return recommender.LGNRecommender(flags_obj, workspace, dm)
        elif flags_obj.model == 'LGNDICE':
            return recommender.LGNDICERecommender(flags_obj, workspace, dm)

    @staticmethod
    def set_device(flags_obj):

        if not flags_obj.use_gpu:
            return torch.device('cpu')
        else:
            return torch.device('cuda:{}'.format(flags_obj.gpu_id))


class VizManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_vm'
        self.exp_name = flags_obj.name
        self.port = flags_obj.port
        self.set_visdom()

    def set_visdom(self):

        self.viz = Visdom(port=self.port, env=self.exp_name)

    def get_new_text_window(self, title):

        win = self.viz.text(title)

        return win

    def append_text(self, text, win):

        self.viz.text(text, win=win, append=True)

    def show_basic_info(self, flags_obj):

        basic = self.viz.text('Basic Information:')
        self.viz.text('Name: {}'.format(flags_obj.name), win=basic, append=True)
        self.viz.text('Model: {}'.format(flags_obj.model), win=basic, append=True)
        self.viz.text('Dataset: {}'.format(flags_obj.dataset), win=basic, append=True)
        self.viz.text('Embedding Size: {}'.format(flags_obj.embedding_size), win=basic, append=True)
        self.viz.text('Initial lr: {}'.format(flags_obj.lr), win=basic, append=True)
        self.viz.text('Batch Size: {}'.format(flags_obj.batch_size), win=basic, append=True)
        self.viz.text('Weight Decay: {}'.format(flags_obj.weight_decay), win=basic, append=True)
        self.viz.text('Negative Sampling Ratio: {}'.format(flags_obj.neg_sample_rate), win=basic, append=True)

        self.basic = basic

        flags = self.viz.text('FLAGS:')
        for flag, value in flags_obj.flag_values_dict().items():
            self.viz.text('{}: {}'.format(flag, value), win=flags, append=True)

        self.flags = flags

    def show_test_info(self, flags_obj):

        test = self.viz.text('Test Information:')
        self.test = test

    def step_update_line(self, title, value):

        if not isinstance(value, numbers.Number):
            return

        if not hasattr(self, title):

            setattr(self, title, self.viz.line([value], [0], opts=dict(title=title)))
            setattr(self, title + '_step', 1)

        else:

            step = getattr(self, title + '_step')
            self.viz.line([value], [step], win=getattr(self, title), update='append')
            setattr(self, title + '_step', step + 1)

    def step_update_multi_lines(self, kv_record):

        for title, value in kv_record.items():
            self.step_update_line(title, value)

    def plot_lines(self, y, x, opts):

        title = opts['title']
        if not hasattr(self, title):
            setattr(self, title, self.viz.line(y, x, opts=opts))
        else:
            self.viz.line(y, x, win=getattr(self, title), opts=opts, update='replace')

    def show_result(self, results, topk):

        self.viz.text('-----topk = {}-----'.format(topk), win=self.test, append=True)
        self.viz.text('-----Results-----', win=self.test, append=True)

        for metric, value in results.items():

            self.viz.text('{}: {}'.format(metric, value), win=self.test, append=True)

        self.viz.text('-----------------', win=self.test, append=True)


class DatasetManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_dm'
        self.make_coo_loader_transformer(flags_obj)
        self.make_npy_loader(flags_obj)
        self.make_csv_loader(flags_obj)

    def make_coo_loader_transformer(self, flags_obj):

        self.coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_transformer = TRANSFORMER.SparseTransformer(flags_obj)

    def make_npy_loader(self, flags_obj):

        self.npy_loader = LOADER.NpyLoader(flags_obj)

    def make_csv_loader(self, flags_obj):

        self.csv_loader = LOADER.CsvLoader(flags_obj)

    def get_dataset_info(self):

        coo_record = self.coo_loader.load(const_util.train_coo_record)

        self.n_user = coo_record.shape[0]
        self.n_item = coo_record.shape[1]

        self.coo_record = coo_record

    def get_skew_dataset(self):

        self.skew_coo_record = self.coo_loader.load(const_util.train_skew_coo_record)

    def get_popularity(self):

        self.popularity = self.npy_loader.load(const_util.popularity)
        return self.popularity

    def get_blend_popularity(self):

        self.blend_popularity = self.npy_loader.load(const_util.blend_popularity)
        return self.blend_popularity


class EarlyStopManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_esm'
        self.min_lr = flags_obj.min_lr
        self.es_patience = flags_obj.es_patience
        self.count = 0
        self.max_metric = 0

    def step(self, lr, metric):

        if lr > self.min_lr:
            if metric > self.max_metric:
                self.max_metric = metric
            return False
        else:
            if metric > self.max_metric:
                self.max_metric = metric
                self.count = 0
                return False
            else:
                self.count = self.count + 1
                if self.count > self.es_patience:
                    return True
                return False


class DICESampler(SAMPLER.Sampler):

    def __init__(self, flags_obj, lil_record, dok_record, neg_sample_rate, popularity, margin=10, pool=10):

        super(DICESampler, self).__init__(flags_obj, lil_record, dok_record, neg_sample_rate)
        self.popularity = popularity
        self.margin = margin
        self.pool = pool

    def adapt(self, epoch, decay):

        self.margin = self.margin*decay

    def generate_negative_samples(self, user, pos_item):

        negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
        mask_type = np.full(self.neg_sample_rate, False, dtype=np.bool)

        user_pos = self.lil_record.rows[user]

        item_pos_pop = self.popularity[pos_item]

        pop_items = np.nonzero(self.popularity > item_pos_pop + self.margin)[0]
        pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
        num_pop_items = len(pop_items)

        unpop_items = np.nonzero(self.popularity < item_pos_pop - 10)[0]
        unpop_items = np.nonzero(self.popularity < item_pos_pop/2)[0]
        unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]
        num_unpop_items = len(unpop_items)

        if num_pop_items < self.pool:
            
            for count in range(self.neg_sample_rate):

                index = np.random.randint(num_unpop_items)
                item = unpop_items[index]
                while item in negative_samples:
                    index = np.random.randint(num_unpop_items)
                    item = unpop_items[index]

                negative_samples[count] = item
                mask_type[count] = False

        elif num_unpop_items < self.pool:
            
            for count in range(self.neg_sample_rate):

                index = np.random.randint(num_pop_items)
                item = pop_items[index]
                while item in negative_samples:
                    index = np.random.randint(num_pop_items)
                    item = pop_items[index]

                negative_samples[count] = item
                mask_type[count] = True
        
        else:

            for count in range(self.neg_sample_rate):

                if np.random.random() < 0.5:

                    index = np.random.randint(num_pop_items)
                    item = pop_items[index]
                    while item in negative_samples:
                        index = np.random.randint(num_pop_items)
                        item = pop_items[index]

                    negative_samples[count] = item
                    mask_type[count] = True

                else:

                    index = np.random.randint(num_unpop_items)
                    item = unpop_items[index]
                    while item in negative_samples:
                        index = np.random.randint(num_unpop_items)
                        item = unpop_items[index]

                    negative_samples[count] = item
                    mask_type[count] = False

        return negative_samples, mask_type

    def sample(self, index):

        user, pos_item = self.get_pos_user_item(index)

        users = np.full(self.neg_sample_rate, user, dtype=np.int64)
        items_pos = np.full(self.neg_sample_rate, pos_item, dtype=np.int64)
        items_neg, mask_type = self.generate_negative_samples(user, pos_item=pos_item)

        return users, items_pos, items_neg, mask_type
