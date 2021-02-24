#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import pandas as pd
import scipy.sparse as sp

import json

import os


class Loader(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_loader'
        self.load_path = flags_obj.load_path
        self.check_load_path()
    
    def check_load_path(self):

        if not os.path.exists(self.load_path):

            print('Error! Load path ({}) does not exist!'.format(self.load_path))
    
    def load(self, filename, **kwargs):

        raise NotImplementedError


class CsvLoader(Loader):

    def __init__(self, flags_obj):

        super(CsvLoader, self).__init__(flags_obj)
    
    def load(self, filename, **kwargs):

        filename = os.path.join(self.load_path, filename)
        record = pd.read_csv(filename, **kwargs)

        return record


class CooLoader(Loader):

    def __init__(self, flags_obj):

        super(CooLoader, self).__init__(flags_obj)
    
    def load(self, filename, **kwargs):

        filename = os.path.join(self.load_path, filename)
        record = sp.load_npz(filename)

        return record


class JsonLoader(Loader):

    def __init__(self, flags_obj):

        super(JsonLoader, self).__init__(flags_obj)
    
    def load(self, filename, **kwargs):

        filename = os.path.join(self.load_path, filename)
        with open(filename, 'r') as f:
            record = json.loads(f.read())
        
        return record


class NpyLoader(Loader):

    def __init__(self, flags_obj):

        super(NpyLoader, self).__init__(flags_obj)
    
    def load(self, filename, **kwargs):

        filename = os.path.join(self.load_path, filename)
        record = np.load(filename)

        return record
