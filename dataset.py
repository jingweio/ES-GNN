from parse import args
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import numpy as np
import scipy.sparse
import scipy.io
import scipy
import json
import csv
import sys

SMALL_OFFSET = 1e-10
DATAPATH = args.DATAPATH + "graph/"


def categorical_label(input):
    r, c = np.where(input == 1)
    labels = np.ones(input.shape[0]) * (-1)
    labels[r] = c
    return labels


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)) + SMALL_OFFSET
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
    # return sparse_to_tuple(features)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


class NCDataset(object):
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(dataname, sub_dataname=''):
    if dataname == 'twitch-e':
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(sub_dataname)
    elif dataname in ('chameleon', 'film', 'squirrel'):
        dataset = load_geom_gcn_dataset(dataname)
    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_citation_dataset(dataname)
    elif dataname in ('polblogs'):
        dataset = load_polblogs()
    elif dataname == "etg_syn_hom":
        if sub_dataname not in ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0"]:
            raise ValueError(f"Invalid dataname: {sub_dataname}")
        dataset = load_etg_syn(sub_dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_twitch(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = "{}twitch/{}".format(DATAPATH, lang)
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)),
                                 (np.array(src), np.array(targ))),
                                shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0]  # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label

    return A, label, features


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATAPATH}{name}.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_citation_dataset(dataset_str):
    # downloaded from https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    """
        Loads input data from gcn/data directory
        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.
        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATAPATH + "citation/", dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATAPATH + "citation_net/", dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    stand_idx_dic = {"tst_idx": idx_test, "trn_idx": idx_train, "val_idx": idx_val}

    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(preprocess_features(features).todense(), dtype=torch.float)
    labels = categorical_label(labels)
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes,
                     "stand_idx_dic": stand_idx_dic}
    labels = torch.tensor(labels, dtype=torch.long)
    dataset.label = labels
    return dataset


def load_polblogs():
    with np.load(f"{DATAPATH}polblogs_dataset/polblogs.npz") as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
        node_feat = torch.tensor(adj_matrix.todense(), dtype=torch.float)  # adj-raw as features
        labels = loader.get('labels')
        num_nodes = len(labels)
        dataset = NCDataset("polblogs")
        dataset.graph = {'edge_index': edge_index,
                         'node_feat': node_feat,
                         'edge_feat': None,
                         'num_nodes': num_nodes}
        labels = torch.tensor(labels, dtype=torch.long)
        dataset.label = labels
        return dataset


def load_etg_syn(hr_level=""):
    data_dic = np.load(f"{DATAPATH}synthetic_dataset/hr_{hr_level}/etg_syn{hr_level}.npy", allow_pickle=True).item()
    edge_index = torch.tensor(data_dic["edge_indexes"], dtype=torch.long)
    labels = torch.tensor(data_dic["labels"], dtype=torch.long)
    num_nodes = len(labels)
    node_feat = torch.tensor(data_dic["feat"], dtype=torch.float)  # guassian noise as feature
    dataset = NCDataset("etg_syn_diff_hr_data")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes,
                     "LR60_splits": data_dic["LR60_splits"],
                     "LR40_splits": data_dic["LR40_splits"],
                     "LR20_splits": data_dic["LR20_splits"]}
    dataset.label = labels
    return dataset
