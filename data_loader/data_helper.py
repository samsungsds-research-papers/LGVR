import numpy as np
import os
import pickle
import torch
from utils.gin_util import S2VGraph


NUM_LABELS = {'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'PROTEINS': 3, 'PTC': 22}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(ds_name):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name]+1), dtype=np.float32)
            labels.append(int(graph_meta[1]))
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0])+1]= 1.
                for k in range(2,len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
            curr_graph = normalize_graph(curr_graph)
            graphs.append(curr_graph)
    graphs = np.array(graphs)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2,0,1])
    return graphs, np.array(labels)


def load_qm9(gin):
    """
    Constructs the graphs and labels of QM9 data set, already split to train, val and test sets
    :return: 6 numpy arrays:
                 train_graphs: N_train,
                 train_labels: N_train x 12
                 val_graphs: N_val,
                 val_labels: N_train x 12
                 test_graphs: N_test,
                 test_labels: N_test x 12
                 each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    train_graphs, train_labels = load_qm9_aux('train', gin)
    val_graphs, val_labels = load_qm9_aux('val', gin)
    test_graphs, test_labels = load_qm9_aux('test', gin)
    return train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels


def load_qm9_aux(which_set, gin):
    """
    Read and construct the graphs and labels of QM9 data set, already split to train, val and test sets
    :param which_set: 'test', 'train' or 'val'
    :return: graphs: (N,)
             labels: N x 12
             each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    base_path = BASE_DIR + "/data/QM9/QM9_{}.p".format(which_set)
    graphs, labels = [], []
    if gin:
        with open(base_path, 'rb') as f:
            data = pickle.load(f)
            for instance in data:
                labels.append(instance['y'])
                nodes_num = instance['usable_features']['x'].shape[0]

                graph = S2VGraph(g=[i for i in range(nodes_num)], label=instance['y'])
                graph.adj_mat = torch.tensor(instance['usable_features']['affinity'])

                # create node features
                node_feat = np.empty((nodes_num, nodes_num, 18))
                for i in range(13):
                    # 13 features per node - for each, create a diag matrix of it as a feature
                    node_feat[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
                node_feat[:, :, 13] = instance['usable_features']['distance_mat']
                node_feat[:, :, 14:] = instance['usable_features']['edge_features']  # shape n x n x 4

                final_node_feat = np.empty((nodes_num, 18))
                for i in range(nodes_num):
                    final_node_feat[i, :] = node_feat[i, i, :]
                graph.node_features = torch.tensor(final_node_feat)

                # create graph neighbors & max_neighbor
                graph.neighbors = []
                graph.max_neighbor = 0
                for i in graph.adj_mat:
                    nbhd = list(np.where(i == 1)[0])
                    graph.neighbors.append(nbhd)
                    graph.max_neighbor = max(len(nbhd), graph.max_neighbor)
                edges = []
                for idx, nbhd in enumerate(graph.neighbors):
                    for n in nbhd:
                        if n > idx:
                            edges.append([idx, n])
                edges.extend([[i, j] for j, i in edges])
                graph.edge_mat = torch.LongTensor(edges).transpose(0, 1)

                graphs.append(graph)
    else:
        with open(base_path, 'rb') as f:
            data = pickle.load(f)
            for instance in data:
                labels.append(instance['y'])
                nodes_num = instance['usable_features']['x'].shape[0]
                graph = np.empty((nodes_num, nodes_num, 19))
                for i in range(13):
                    # 13 features per node - for each, create a diag matrix of it as a feature
                    graph[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
                graph[:, :, 13] = instance['usable_features']['distance_mat']
                graph[:, :, 14] = instance['usable_features']['affinity']
                graph[:, :, 15:] = instance['usable_features']['edge_features']  # shape n x n x 4
                graphs.append(graph)
        graphs = np.array(graphs)
        for i in range(graphs.shape[0]):
            graphs[i] = np.transpose(graphs[i], [2, 0, 1])

    labels = np.array(labels).squeeze()  # shape N x 12

    return graphs, labels


def get_train_val_indexes(num_val, ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param num_val: number of the split
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/10fold_idx".format(ds_name)
    train_file = "train_idx-{0}.txt".format(num_val)
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "test_idx-{0}.txt".format(num_val)
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def group_same_size(graphs, labels, gin):
    """
    group graphs of same size to same array
    :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs adjacency matrix
    :param labels: numpy array of labels
    :return: two numpy arrays. graphs arrays in the shape (num of different size graphs) where each entry is a numpy array
            in the shape (number of graphs with this size, num vertex, num. vertex, num vertex labels)
            the second arrayy is labels with correspons shape
    """
    if gin:
        sizes = list(map(lambda t: t.node_features.shape[0], graphs))
    else:
        sizes = list(map(lambda t: t.shape[1], graphs))
    indexes = np.argsort(sizes)
    graphs = np.array(graphs)[indexes]
    labels = labels[indexes]
    r_graphs = []
    r_labels = []
    one_size = []
    start = 0
    if gin:
        size = graphs[0].node_features.shape[0]
        for i in range(len(graphs)):
            if graphs[i].node_features.shape[0] == size:
                one_size.append(graphs[i])
            else:
                r_graphs.append(one_size)
                r_labels.append(np.array(labels[start:i]))
                start = i
                one_size = []
                size = graphs[i].node_features.shape[0]
                one_size.append(graphs[i])
        r_graphs.append(one_size)
        r_labels.append(np.array(labels[start:]))
    else:
        size = graphs[0].shape[1]
        for i in range(len(graphs)):
            if graphs[i].shape[1] == size:
                one_size.append(np.expand_dims(graphs[i], axis=0))
            else:
                r_graphs.append(np.concatenate(one_size, axis=0))
                r_labels.append(np.array(labels[start:i]))
                start = i
                one_size = []
                size = graphs[i].shape[1]
                one_size.append(np.expand_dims(graphs[i], axis=0))
        r_graphs.append(np.concatenate(one_size, axis=0))
        r_labels.append(np.array(labels[start:]))
    return r_graphs, r_labels


# helper method to shuffle each same size graphs array
def shuffle_same_size(graphs, labels):
    r_graphs, r_labels = [], []
    for i in range(len(labels)):
        curr_graph, curr_labels = shuffle(graphs[i], labels[i])
        r_graphs.append(curr_graph)
        r_labels.append(curr_labels)
    return r_graphs, r_labels


def split_to_batches(graphs, labels, size, gin):
    """
    split the same size graphs array to batches of specified size
    last batch is in size num_of_graphs_this_size % size
    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param size: batch size
    :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
                corresponds labels
    """
    r_graphs = []
    r_labels = []
    if gin:
        for k in range(len(graphs)):
            r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])
            r_graphs = r_graphs + np.split(np.array(graphs[k]), [j for j in range(size, len(graphs[k]), size)])
        ret1 = r_graphs
        ret2 = np.empty(len(r_labels), dtype=object)
        ret2[:] = r_labels
    else:
        for k in range(len(graphs)):
            r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
            r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])
        # Avoid bug for batch_size=1, where instead of creating numpy array of objects, we had numpy array of floats with
        # different sizes - could not reshape
        ret1, ret2 = np.empty(len(r_graphs), dtype=object), np.empty(len(r_labels), dtype=object)
        ret1[:] = r_graphs
        ret2[:] = r_labels
    return ret1, ret2


# helper method to shuffle the same way graphs and labels arrays
def shuffle(graphs, labels):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    np.random.shuffle(shf)
    return np.array(graphs)[shf], labels[shf]


def normalize_graph(curr_graph):
    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
    return np.add(spred_adj, labels)

