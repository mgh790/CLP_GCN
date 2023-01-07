import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import pickle
import os
flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_randomdata(dataset_str, train_size, validation_size,  shuffle=True,aplus=None):
    """Load data."""

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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
    # features = sp.eye(features.shape[0]).tolil()
    # features = sp.lil_matrix(allx)
    if aplus!=None:
        #################### weighted matrix ##################################################3
        fg = nx.Graph()
        fg.add_nodes_from(list(range(len(graph))))
        fg2 = nx.Graph()
        fg2.add_nodes_from(list(range(len(graph))))
        if FLAGS.a_plus == 'Salton':
            edges = set((a, b, len(set(graph[a]) & set(graph[b])) / len(graph[b])) for a in graph
                        # edges = set((a, b, len(set(graph[a]) & set(graph[b])) ) for a in graph
                        for b in sum([graph[c] for c in graph[a]], []) if len(set(graph[a]) & set(graph[b])) > 1)
        fg.add_weighted_edges_from(edges)
        adj2 = nx.adjacency_matrix(fg)
        adjW=adj2+adj
    else:
        adjW=adj


    labels = np.vstack((ally, ty))

    if dataset_str.startswith('nell'):
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/planetoid/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended, dtype=np.float32)
            print("Done!")
            save_sparse_csr("data/planetoid/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/planetoid/{}.features.npz".format(dataset_str))

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    global all_labels
    all_labels = labels.copy()

    # split the data set
    idx = np.arange(len(labels))
    no_class = labels.shape[1]  # number of class
    # validation_size = validation_size * len(idx) // 100
    # if not hasattr(train_size, '__getitem__'):
    train_size = [train_size for i in range(labels.shape[1])]
    if shuffle:
        np.random.shuffle(idx)
    idx_train = []
    count = [0 for i in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(no_class):
            if labels[i, j] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    test_size =1000 #model_config['test_size']
    # validate=True
    validate=""
    if validate==True:
        idx_val = idx[next:next+validation_size]
        assert next+validation_size+test_size < len(idx)
        idx_test = idx[-test_size:] if test_size else idx[next+validation_size:]
    else:
        if test_size:
            assert next+test_size<len(idx)
        idx_val = idx[-test_size:] if test_size else idx[next:]
        idx_test = idx[-test_size:] if test_size else idx[next:]
    # else:
    #     labels_of_class = [0]
    #     while (np.prod(labels_of_class) == 0):
    #         np.random.shuffle(idx)
    #         idx_train = idx[0:int(len(idx) * train_size // 100)]
    #         labels_of_class = np.sum(labels[idx_train], axis=0)
    #     idx_val = idx[-500 - validation_size:-500]
    #     idx_test = idx[-500:]
    print('labels of each class : ', np.sum(labels[idx_train], axis=0))
    # idx_val = idx[len(idx) * train_size // 100:len(idx) * (train_size // 2 + 50) // 100]
    # idx_test = idx[len(idx) * (train_size // 2 + 50) // 100:len(idx)]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    ## yalda added
    unlbl_idx=[i for i in range(adj.shape[0]) if (train_mask[i]==False) and (test_mask[i]==False) and (val_mask[i]==False)]
    unlbl_idx=np.asarray(unlbl_idx)
    unlbl_mask = sample_mask(unlbl_idx, labels.shape[0])
    y_unlbl = np.zeros(labels.shape)
    y_unlbl[unlbl_mask, :] = labels[unlbl_mask, :]

    # else:
    #     idx_test = test_idx_range.tolist()
    #     idx_train = range(len(y))
    #     idx_val = range(len(y), len(y) + 500)
    #
    #     train_mask = sample_mask(idx_train, labels.shape[0])
    #     val_mask = sample_mask(idx_val, labels.shape[0])
    #     test_mask = sample_mask(idx_test, labels.shape[0])
    #
    #     y_train = np.zeros(labels.shape)
    #     y_val = np.zeros(labels.shape)
    #     y_test = np.zeros(labels.shape)
    #     y_train[train_mask, :] = labels[train_mask, :]
    #     y_val[val_mask, :] = labels[val_mask, :]
    #     y_test[test_mask, :] = labels[test_mask, :]

    size_of_each_class = np.sum(labels[idx_train], axis=0)
    return adj,adjW, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,unlbl_mask

def load_data(dataset_str,aplus=None):
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
    ind.dataset_str.test.in'dex => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    if 'nell.0' in dataset_str:
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack(
                (features, sp.lil_matrix(
                    (features.shape[0], len(isolated_node_idx))
                )), dtype=np.float32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(
                len(isolated_node_idx), dtype=np.float32)
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str),
                            features)
        else:
            features = load_sparse_csr(
                "data/{}.features.npz".format(dataset_str))

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    if aplus!=None:
        #################### weighted matrix ##################################################3
        fg = nx.Graph()
        fg.add_nodes_from(list(range(len(graph))))
        fg2 = nx.Graph()
        fg2.add_nodes_from(list(range(len(graph))))
        if FLAGS.a_plus == 'Salton':
            edges = set((a, b, len(set(graph[a]) & set(graph[b])) / len(graph[b])) for a in graph
                        # edges = set((a, b, len(set(graph[a]) & set(graph[b])) ) for a in graph
                        for b in sum([graph[c] for c in graph[a]], []) if len(set(graph[a]) & set(graph[b])) > 1)

        fg.add_weighted_edges_from(edges)
        adj2 = nx.adjacency_matrix(fg)
        adj_W=adj2+adj
    else:
        adj_W=adj

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    #

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    unlbl_idx=[i for i in range(adj.shape[0]) if (train_mask[i]==False) and (test_mask[i]==False) and (val_mask[i]==False)]
    unlbl_idx=np.asarray(unlbl_idx)
    unlbl_mask = sample_mask(unlbl_idx, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_unlbl = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    y_unlbl[unlbl_mask, :] = labels[unlbl_mask, :]

    return adj,adj_W, features, y_train, y_val, y_test,y_unlbl,\
           train_mask, val_mask, test_mask,unlbl_mask,labels,idx_train,idx_val,idx_test

def save_sparse_csr(filename,array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix(
        (loader['data'], loader['indices'], loader['indptr']),
        shape=loader['shape']
    )
def cal_degree(adjD):
        diagonal_degrees = []
        for row in adjD:
            diagonal_degrees.append(np.sum(row!=0))

## embeing for nell dataset
from sklearn import preprocessing
def load_embedding(model_type, dataset_str, all_embedding=False):
    fname = "data/{}.{}.emb".format(model_type, dataset_str)
    if all_embedding:
        fname += 'all'
    print(fname)
    with open(fname, 'rb') as f:
        if sys.version_info > (3, 0):
            saved_embeddings = pkl.load(f, encoding='latin1')
        else:
            saved_embeddings = pkl.load(f)
    print(saved_embeddings.shape)
    return preprocessing.normalize(saved_embeddings, norm='l2')


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
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support,labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
