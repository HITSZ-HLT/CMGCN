# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='context_indices', shuffle=True, sort=False,opt = None):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.opt = opt

        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)
        

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches


    def pad_data(self, batch_data):
        batch_label = []
        batch_graph = []
        batch_box_vit = []
        batch_bert_indices = []
        batch_box_indices = []

        if True:
            bert_indices_max_len = max([len(t['bert_indices']) for t in batch_data])
            box_indices_max_len = max([sum([len(x_) for x_ in t["box_indices"]]) for t in batch_data])
            vit_size = max([len(t["box_vit"]) for t in batch_data])

            for item in batch_data:
                label,graph,box_vit,box_indices,bert_indices = \
                    item['label'],item['graph'],item['box_vit'],item['box_indices'],item['bert_indices']

                image_graph = item['image_graph']


                if graph.shape[0] < bert_indices_max_len:
                    graph = numpy.pad(graph, ((0,bert_indices_max_len-graph.shape[0]),\
                        (0,bert_indices_max_len-graph.shape[0])), 'constant')
                
                
                graph = numpy.pad(graph,((0,10),(0,10)),'constant')
                #graph = numpy.pad(graph,((0,5),(0,5)),'constant')

                for i in range(image_graph.shape[0]-2):
                    for j in range(image_graph.shape[1]):
                        if not numpy.isnan(image_graph[i][j]):
                            graph[i+1][j+bert_indices_max_len] = image_graph[i][j] + 1
                            graph[j+bert_indices_max_len][i+1] = image_graph[i][j] + 1
                        else:
                            graph[i+1][j+bert_indices_max_len] =  1
                            graph[j+bert_indices_max_len][i+1] =  1
                
                for i in range(image_graph.shape[1]):
                    graph[i+bert_indices_max_len][i+bert_indices_max_len] = 1
                

                new_box_indices = []
                for box_indice in box_indices:
                    if len(box_indice) < box_indices_max_len:
                        box_indice = box_indice + [0]*(box_indices_max_len-len(box_indice))
                    new_box_indices.append(numpy.array(box_indice))
                
                while len(new_box_indices) < 10:
                    new_box_indices.append([0]*box_indices_max_len)

                new_box_indices = numpy.array(new_box_indices)

                batch_graph.append(graph)
                batch_bert_indices.append(numpy.pad(bert_indices,(0,bert_indices_max_len - len(bert_indices)),'constant'))
                batch_box_indices.append(new_box_indices)
                batch_label.append(label)
                t = [x.numpy() for x in box_vit]
                while len(t) < 10:
                    t.append(numpy.zeros(768))

                batch_box_vit.append(torch.tensor(t))

            return {
                'label': torch.tensor(batch_label),
                'graph':torch.tensor(batch_graph),
                'box_vit':torch.tensor([x.numpy() for x in batch_box_vit]),
                'bert_indices':torch.tensor(batch_bert_indices),
                'box_indices':torch.tensor(batch_box_indices)
                }


    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
