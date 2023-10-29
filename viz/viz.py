#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import torch
import numpy as np
from sklearn.manifold import TSNE
import time


class Vizer(object):

    def __init__(self):

        self.name = 'vizer'

    def tsne(self, n_components, embedding, filename):

        start_time = time.time()
        print('start TSNE! Grab a cup of coffee :)')

        embedding_tsne = TSNE(n_components=n_components).fit_transform(embedding)

        np.savetxt(filename, embedding_tsne)

        print('TSNE costs time: {:.2f} s.'.format(time.time() - start_time))

        return embedding_tsne


class ParameterManager(object):

    def __init__(self):

        self.name = 'pm'

    def get_item_embedding(self, path):

        model = torch.load(path)
        items_int = model['items_int'].cpu().numpy()
        items_pop = model['items_pop'].cpu().numpy()
        embeddings = np.vstack([items_int, items_pop])

        items_int_hue = ['item_int' for _ in range(len(items_int))]
        items_pop_hue = ['item_pop' for _ in range(len(items_pop))]
        hue = items_int_hue + items_pop_hue

        return embeddings, hue

    def get_item_pop_embedding(self, path):

        model = torch.load(path)
        items_pop = model['items_pop'].cpu().numpy()
        embeddings = items_pop

        return embeddings, None


def visualize():

    pm = ParameterManager()
    vizer = Vizer()

    path = '/path/to/ckpt/epoch_xxx.pth'

    embedding, _ = pm.get_item_embedding(path)
    filename = './data/xxx.txt'
    _ = vizer.tsne(2, embedding, filename)

    pop_embedding, _ = pm.get_item_pop_embedding(path)
    filename = './data/xxx_xxx.txt'
    _ = vizer.tsne(2, pop_embedding, filename)


if __name__ == "__main__":

    visualize()

