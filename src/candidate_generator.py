#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from collections.abc import Iterable

import numpy as np
import torch

import faiss


class CandidateGenerator(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_cg'

    def generate(self, user, k):

        raise NotImplementedError


class RandomGenerator(CandidateGenerator):

    def __init__(self, flags_obj, items):

        super(RandomGenerator, self).__init__(flags_obj)
        self.items = items
        self.n_item = len(items)

    def generate(self, users, k):

        if not isinstance(users, Iterable):
            return self.choice(k)
        items_chosen = [self.choice(k) for _ in users]
        return np.stack(items_chosen, axis=0)

    def choice(self, k):

        item_chosen = np.full(k, -1)
        for count in range(k):
            i = np.random.randint(self.n_item)
            while i in item_chosen:
                i = np.random.randint(self.n_item)
            item_chosen[count] = i
        return item_chosen


class PopularityGenerator(CandidateGenerator):

    def __init__(self, flags_obj, popularity, max_k):

        super(PopularityGenerator, self).__init__(flags_obj)
        self.popularity = popularity
        self.max_k = max_k
        self.get_popular_items()

    def get_popular_items(self):

        popularity_tensor = torch.LongTensor(self.popularity)
        self.popular_items = torch.topk(popularity_tensor, self.max_k)[1].numpy()

    def generate(self, user, k):

        if not isinstance(user, Iterable):
            return self.popular_items[:k]
        items = [self.popular_items[:k] for _ in user]
        return np.stack(items, axis=0)


class FaissInnerProductMaximumSearchGenerator(CandidateGenerator):

    def __init__(self, flags_obj, items):

        super(FaissInnerProductMaximumSearchGenerator, self).__init__(flags_obj)
        self.items = items
        self.embedding_size = items.shape[1]
        self.make_index(flags_obj)

    def make_index(self, flags_obj):

        self.make_index_brute_force(flags_obj)

        if flags_obj.cg_use_gpu:

            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, flags_obj.cg_gpu_id, self.index)

    def make_index_brute_force(self, flags_obj):

        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(self.items)

    def generate(self, users, k):

        _, I = self.index.search(users, k)

        return I

    def generate_with_distance(self, users, k):

        D, I = self.index.search(users, k)

        return D, I
