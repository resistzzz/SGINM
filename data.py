# -*- coding: utf-8 -*-

import torch
import itertools
import random
import copy
import networkx as nx
import numpy as np

# Default word tokens
PAD_token = 0  # Used for padding short sentences

class Data(object):

    def __init__(self, data_pairs, status='train'):
        self.data_pairs = data_pairs
        self.data_length = len(self.data_pairs)
        self.status = status
        self.num_words = 1
        self.word2index = {}
        self.index2word = {PAD_token: 'PAD'}
        if status == 'all':
            self.travel_around_data()

    def travel_around_data(self):
        for seq, lab in self.data_pairs:
            seq = seq + [lab]
            for s in seq:
                if s not in self.word2index:
                    self.word2index[s] = self.num_words
                    self.index2word[self.num_words] = s
                    self.num_words += 1

    def zeroPadding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence]

    def generate_batch_slices(self, batch_size):
        n_batch = int(self.data_length / batch_size)
        if self.data_length % batch_size != 0:
            n_batch += 1
        slices = [0] * n_batch
        for i in range(n_batch):
            if i != n_batch - 1:
                slices[i] = self.data_pairs[i*batch_size:(i+1)*batch_size]
            else:
                slices[i] = self.data_pairs[i*batch_size:]
        return slices

    def inputVar(self, l):
        indexes_batch = [self.indexesFromSentence(sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

        # mask0, mask1 = [], []   # mask0标记是否填充, mask1标记填充前序列长度, mask_inf在Padding的位置标记为-inf
        # max_len = len(indexes_batch[0])
        # mask_inf = np.full((len(indexes_batch), max_len), float('-inf'), dtype=np.float32)
        # for seq in indexes_batch:
        #     mask0.append([0] * max_len)
        #     for i in range(len(seq)):
        #         mask0[-1][i] = 1
        #     mask1.append(len(seq))

        max_len = len(indexes_batch[0])
        mask0 = np.zeros((len(indexes_batch), max_len), dtype=np.float32)
        mask1 = []
        mask_inf = np.full((len(indexes_batch), max_len), float('-inf'), dtype=np.float32)
        for i in range(len(indexes_batch)):
            mask0[i, :len(indexes_batch[i])] = 1.0
            mask1.append(len(indexes_batch[i]))
            mask_inf[i, :len(indexes_batch[i])] = 0.0

        indexes_pad = self.zeroPadding(indexes_batch)
        indexes_var = torch.LongTensor(indexes_pad)

        mask0_var = torch.FloatTensor(mask0)
        mask1_var = torch.FloatTensor(mask1)
        maskinf_var = torch.FloatTensor(mask_inf)

        return indexes_var, lengths, mask0_var, mask1_var, maskinf_var

    def outputVar(self, l):
        indexes_l = [self.word2index[word] for word in l]
        labelVar = torch.LongTensor(indexes_l)
        return labelVar

    def batch2TrainData(self, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp_var, lengths, mask0, mask1, mask_inf = self.inputVar(input_batch)
        out_var = self.outputVar(output_batch)

        return inp_var, lengths, mask0, mask1, mask_inf, out_var

def split_short_long(data_pairs, thred=5):
    short_pairs = []
    long_pairs = []
    for seq, lab in data_pairs:
        if len(seq) <= thred:
            short_pairs.append((seq, lab))
        else:
            long_pairs.append((seq, lab))
    print('Short session: %d, %0.2f\tLong session: %d, %0.2f' %
          (len(short_pairs), 100.0 * len(short_pairs) / len(data_pairs), len(long_pairs), 100.0 * len(long_pairs) / len(data_pairs)))
    return short_pairs, long_pairs


# #############################################################
# Virtual CF data
def build_graph(train_seqs):
    graph = nx.DiGraph()
    for seq in train_seqs:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    node_outdegree = {}
    # get node outdegree and 归一化边的权重
    for node in graph.nodes:
        sum = 0
        for j, i in graph.out_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for k, v in graph.out_edges(node):
                graph.add_edge(k, v, weight=graph.get_edge_data(k, v)['weight'] / sum)
        node_outdegree[node] = sum
    return graph, node_outdegree


def trucated_random_walk_generate_data(graph, node_outdgree, train_data, num=10000, max_length=6, alpha=0.1):
    nodes = graph.nodes
    adj_dict = dict(graph.adjacency())
    for node in adj_dict:
        node_adjs = []
        weights = []
        for n in adj_dict[node]:
            node_adjs.append(n)
            weights.append(adj_dict[node][n]['weight'])
        adj_dict[node] = (node_adjs, weights)
    # 处理节点出度，将其归一化成概率
    nodes_outdegree_list = list(node_outdgree.items())
    nodes_outdegree_list = sorted(nodes_outdegree_list, key=lambda x: x[1], reverse=True)
    nodes = [n[0] for n in nodes_outdegree_list]
    outdegree = [n[1] for n in nodes_outdegree_list]
    outdegree = np.array(outdegree) / sum(outdegree)

    outpath = []
    for j in range(num):
        flag = 0
        node = np.random.choice(nodes, size=1, p=outdegree)[0]
        path = [node]
        max_length = np.random.randint(2, 10)
        for i in range(max_length - 1):
            a = adj_dict[path[-1]][0]
            b = adj_dict[path[-1]][1]
            if len(a) != 0 and len(b) != 0:
                next_node = get_next_node(adj_dict[path[-1]][0], adj_dict[path[-1]][1])
                path.append(next_node)
            else:
                flag = 1
                break
            tmp = random.uniform(0, 1)
            if tmp < alpha:
                break
        if flag == 0:
            outpath.append(path)
    all_pairs = []
    # all_pairs = process_seqs(outpath)
    for path in outpath:
        pair = (path[:-1], path[-1])
        all_pairs.append(pair)
    # print('VCF中没有在原数据集中出现的数目: %d, 比例：%0.4f' % (len(all_pairs), len(all_pairs) / len(outpath)))

    # #######################
    # 测试只使用VCF，丢弃真实数据集，给VCF做个增强
    # all_pairs = process_seqs(outpath)
    # print('len vcf: ', len(all_pairs))
    # #######################

    all_pairs += train_data
    random.shuffle(all_pairs)
    return all_pairs

def get_next_node(nodes, p):
    # print(nodes, p)
    p = np.array(p)
    if sum(p) != 1:
        p = p / sum(p)
    next_node = np.random.choice(nodes, size=1, p=p)[0]

    return next_node

def plus_vcf_data(train_data, num=10000):
    # 默认使用train_pairs的列表
    tr_seqs = []
    for seq, lab in train_data:
        tr_seqs.append(seq + [lab])

    graph, node_outdgree = build_graph(tr_seqs)
    # print(len(graph.nodes()))
    all_pairs = trucated_random_walk_generate_data(graph, node_outdgree, train_data, num=num)
    return all_pairs

# ###############################################################








