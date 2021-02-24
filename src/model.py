#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import dgl.function as fn

import utils

from deprecated import deprecated
from tqdm import tqdm

import random


class MF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(MF, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)

    def pair_forward(self, user, item_p, item_n):

        user = self.users[user]
        item_p = self.items[item_p]
        item_n = self.items[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def point_forward(self, user, item):

        user = self.users[user]
        item = self.items[item]

        score = torch.sum(user * item, 2)

        return score

    def get_item_embeddings(self):

        return self.items.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        return self.users.detach().cpu().numpy().astype('float32')


class LGConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None):
        super(LGConv, self).__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm

    def forward(self, graph, feat):

        graph = graph.local_var()
        if self._cached_h is not None:
            feat = self._cached_h
        else:
            # compute normalization
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)
            # compute (D^-1 A^k D)^k X
            for _ in range(self._k):
                feat = feat * norm
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm

            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat

        return feat


class LGN(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout):

        super(LGN, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.embeddings = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings.size(1))
        self.embeddings.data.uniform_(-stdv, stdv)

    def pair_forward(self, user, item_p, item_n, graph, training=True):

        features = [self.embeddings]
        h = self.embeddings
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features.append(h)

        features = torch.stack(features, dim=2)
        features = torch.mean(features, dim=2)

        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        user = features[user]
        item_p = features[item_p]
        item_n = features[item_n]

        p_score = torch.sum(user * item_p, 2)
        n_score = torch.sum(user * item_n, 2)

        return p_score, n_score

    def get_embeddings(self, graph):

        features = [self.embeddings]
        h = self.embeddings
        for layer in self.layers:
            h = layer(graph, h)
            features.append(h)

        features = torch.stack(features, dim=2)
        features = torch.mean(features, dim=2)

        users = features[:self.n_user]
        items = features[self.n_user:]

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')


class DICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, dis_loss, dis_pen, int_weight, pop_weight):

        super(DICE, self).__init__()

        self.users_int = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.users_pop = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_int = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_pop = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.int_weight = int_weight
        self.pop_weight = pop_weight

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight*decay
        self.pop_weight = self.pop_weight*decay

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):

        stdv = 1. / math.sqrt(self.users_int.size(1))
        self.users_int.data.uniform_(-stdv, stdv)
        self.users_pop.data.uniform_(-stdv, stdv)
        self.items_int.data.uniform_(-stdv, stdv)
        self.items_pop.data.uniform_(-stdv, stdv)

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask*torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask):

        users_int = self.users_int[user]
        users_pop = self.users_pop[user]
        items_p_int = self.items_int[item_p]
        items_p_pop = self.items_pop[item_p]
        items_n_int = self.items_int[item_n]
        items_n_pop = self.items_pop[item_n]

        p_score_int = torch.sum(users_int*items_p_int, 2)
        n_score_int = torch.sum(users_int*items_n_int, 2)

        p_score_pop = torch.sum(users_pop*items_p_pop, 2)
        n_score_pop = torch.sum(users_pop*items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = self.items_int[item_all]
        item_pop = self.items_pop[item_all]
        user_all = torch.unique(user)
        user_int = self.users_int[user_all]
        user_pop = self.users_pop[user_all]
        discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

        loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

        return loss

    def get_item_embeddings(self):

        item_embeddings = torch.cat((self.items_int, self.items_pop), 1)
        #item_embeddings = self.items_pop
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        user_embeddings = torch.cat((self.users_int, self.users_pop), 1)
        #user_embeddings = self.users_pop
        return user_embeddings.detach().cpu().numpy().astype('float32')


class LGNDICE(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout, dis_loss, dis_pen, int_weight, pop_weight):

        super(LGNDICE, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.int_weight = int_weight
        self.pop_weight = pop_weight

        self.embeddings_int = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_pop = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings_int.size(1))
        self.embeddings_int.data.uniform_(-stdv, stdv)
        self.embeddings_pop.data.uniform_(-stdv, stdv)

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight*decay
        self.pop_weight = self.pop_weight*decay

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask*torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask, graph, training=True):

        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_int.append(h)

        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_pop.append(h)

        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)

        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        users_int = features_int[user]
        users_pop = features_pop[user]
        items_p_int = features_int[item_p]
        items_p_pop = features_pop[item_p]
        items_n_int = features_int[item_n]
        items_n_pop = features_pop[item_n]

        p_score_int = torch.sum(users_int*items_p_int, 2)
        n_score_int = torch.sum(users_int*items_n_int, 2)

        p_score_pop = torch.sum(users_pop*items_p_pop, 2)
        n_score_pop = torch.sum(users_pop*items_n_pop, 2)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop

        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)
        loss_pop = self.mask_bpr_loss(n_score_pop, p_score_pop, mask) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        item_all = torch.unique(torch.cat((item_p, item_n)))
        item_int = features_int[item_all]
        item_pop = features_pop[item_all]
        user_all = torch.unique(user)
        user_int = features_int[user_all]
        user_pop = features_pop[user_all]
        discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

        loss = self.int_weight*loss_int + self.pop_weight*loss_pop + loss_total - self.dis_pen*discrepency_loss

        return loss

    def get_embeddings(self, graph):

        features_int = [self.embeddings_int]
        h = self.embeddings_int
        for layer in self.layers:
            h = layer(graph, h)
            features_int.append(h)

        features_int = torch.stack(features_int, dim=2)
        features_int = torch.mean(features_int, dim=2)

        users_int = features_int[:self.n_user]
        items_int = features_int[self.n_user:]

        features_pop = [self.embeddings_pop]
        h = self.embeddings_pop
        for layer in self.layers:
            h = layer(graph, h)
            features_pop.append(h)

        features_pop = torch.stack(features_pop, dim=2)
        features_pop = torch.mean(features_pop, dim=2)
        users_pop = features_pop[:self.n_user]
        items_pop = features_pop[self.n_user:]

        items = torch.cat((items_int, items_pop), 1)
        users = torch.cat((users_int, users_pop), 1)

        return items.detach().cpu().numpy().astype('float32'), users.detach().cpu().numpy().astype('float32')

