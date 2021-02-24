#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

import data
import model
import utils
import candidate_generator as cg
import config.const as const_util

import os

import numpy as np
import dgl

import data_utils.loader as LOADER


class Recommender(object):

    def __init__(self, flags_obj, workspace, dm):

        self.dm = dm
        self.model_name = flags_obj.model
        self.flags_obj = flags_obj
        self.set_device()
        self.set_model()
        self.workspace = workspace

    def set_device(self):

        self.device  = utils.ContextManager.set_device(self.flags_obj)

    def set_model(self):

        raise NotImplementedError

    def transfer_model(self):

        self.model = self.model.to(self.device)

    def save_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        torch.save(self.model.state_dict(), model_path)

    def load_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def get_dataloader(self):

        raise NotImplementedError

    def get_pair_dataloader(self):

        raise NotImplementedError

    def get_point_dataloader(self):

        raise NotImplementedError

    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.flags_obj.lr, weight_decay=self.flags_obj.weight_decay, betas=(0.5, 0.99), amsgrad=True)

    def inference(self, sample):

        raise NotImplementedError

    def make_cg(self):

        raise NotImplementedError

    def cg(self, users, topk):

        raise NotImplementedError


class MFRecommender(Recommender):

    def __init__(self, flags_obj, workspace, dm):

        super(MFRecommender, self).__init__(flags_obj, workspace, dm)
        self.dm.get_skew_dataset()

    def set_model(self):

        self.model = model.MF(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size)

    def get_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_blend_pair_dataloader(self.flags_obj, self.dm)

    def pair_inference(self, sample):

        user, item_p, item_n = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)

        p_score, n_score = self.model.pair_forward(user, item_p, item_n)

        return p_score, n_score

    def make_cg(self):

        self.item_embeddings = self.model.get_item_embeddings()
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)

        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users, topk):

        return self.generator.generate(self.user_embeddings[users], topk)


class LGNRecommender(MFRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(LGNRecommender, self).__init__(flags_obj, workspace, dm)
        self.init_graph(flags_obj)

    def init_graph(self, flags_obj):

        coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_adj_graph = coo_loader.load(const_util.train_blend_coo_adj_graph)

        self.graph = dgl.DGLGraph()

        num_nodes = self.coo_adj_graph.shape[0]
        self.graph.add_nodes(num_nodes)
        self.graph.ndata['feature'] = torch.arange(num_nodes)

        self.graph.add_edges(self.coo_adj_graph.row, self.coo_adj_graph.col)
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())

        self.graph.readonly()

    def set_model(self):

        self.model = model.LGN(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.num_layers, self.flags_obj.dropout)

    def pair_inference(self, sample):

        user, item_p, item_n = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)

        p_score, n_score = self.model.pair_forward(user, item_p, item_n, self.graph)

        return p_score, n_score

    def make_cg(self):

        self.item_embeddings, self.user_embeddings = self.model.get_embeddings(self.graph)
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)


class DICERecommender(MFRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(DICERecommender, self).__init__(flags_obj, workspace, dm)

    def set_model(self):

        self.model = model.DICE(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.dis_loss, self.flags_obj.dis_pen, self.flags_obj.int_weight, self.flags_obj.pop_weight)

    def get_pair_dataloader(self):

        return data.FactorizationDataProcessor.get_DICE_dataloader(self.flags_obj, self.dm)

    def get_loss(self, sample):

        user, item_p, item_n, mask = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)
        mask = mask.to(self.device)

        loss = self.model(user, item_p, item_n, mask)

        return loss

    def make_cg(self):

        self.item_embeddings = self.model.get_item_embeddings()
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)

        self.user_embeddings = self.model.get_user_embeddings()

    def cg(self, users, topk):

        return self.generator.generate(self.user_embeddings[users], topk)

    def adapt(self, epoch, decay):

        self.model.adapt(epoch, decay)


class LGNDICERecommender(DICERecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(LGNDICERecommender, self).__init__(flags_obj, workspace, dm)
        self.init_graph(flags_obj)

    def init_graph(self, flags_obj):

        coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_adj_graph = coo_loader.load(const_util.train_blend_coo_adj_graph)

        self.graph = dgl.DGLGraph()

        num_nodes = self.coo_adj_graph.shape[0]
        self.graph.add_nodes(num_nodes)
        self.graph.ndata['feature'] = torch.arange(num_nodes)

        self.graph.add_edges(self.coo_adj_graph.row, self.coo_adj_graph.col)
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())

        self.graph.readonly()

    def set_model(self):

        self.model = model.LGNDICE(self.dm.n_user, self.dm.n_item, self.flags_obj.embedding_size, self.flags_obj.num_layers, self.flags_obj.dropout, self.flags_obj.dis_loss, self.flags_obj.dis_pen, self.flags_obj.int_weight, self.flags_obj.pop_weight)

    def get_loss(self, sample):

        user, item_p, item_n, mask = sample

        user = user.to(self.device)
        item_p = item_p.to(self.device)
        item_n = item_n.to(self.device)
        mask = mask.to(self.device)

        loss = self.model(user, item_p, item_n, mask, self.graph)

        return loss

    def make_cg(self):

        self.item_embeddings, self.user_embeddings = self.model.get_embeddings(self.graph)
        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, self.item_embeddings)

    def adapt(self, epoch, decay):

        self.model.adapt(epoch, decay)
