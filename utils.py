from collections import Counter
import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt


def convert_labels_to_ids(con_types):
    """ Maps names of the connection types to corresponding IDs
        and returns a one-hot vector to be used in computation.

    Args:
        con_types (set): set of connection types

    Returns:
        ids (list): one-hot list of ids corresponding to connection types
    """
    label_dict = {
        '-': 0, 'smurf': 1, 'neptune': 2, 'teardrop': 3, 'portsweep': 4,
        'ipsweep': 5, 'back': 6, 'satan': 7, 'nmap': 8, 'warezclient': 9,
        'pod': 10, 'rootkit': 11, 'apache2': 12, 'snmpguess': 13, 'snmpgetattack': 14,
        'processtable': 15, 'smurfttl': 16, 'warez': 17, 'guest': 18, 'dict': 19,
        'mscan': 20, 'httptunnel-e': 21, 'mailbomb': 22, 'ignore': 23, 'saint': 24
    }
    ids = [0] * len(label_dict)

    # empty valid_answer file means '-'
    if not con_types:
        ids[0] = 1
        return ids

    for con_type in con_types:
        ids[label_dict[con_type]] = 1

    return ids


def update_feature_matrix(feature_matrix, node_dict):
    """ Given an empty matrix and all nodes' information,
        assign values to the matrix.

    Args:
        feature_matrix (torch.Tensor): empty matrix of shape (num_nodes, num_features)
        node_dict (dictionary): dictionary of dictionaries containing node info per node

    Returns:
        feature_matrix (torch.Tensor): updated matrix
    """
    for node, features in node_dict.items():
        out_degree = len(features["out_neighbors"])
        in_degree = len(features["in_neighbors"])
        connect_freq = features["connect_freq"]
        connected_freq = features["connected_freq"]
        most_freq_port_num = int(Counter(features["port_nums"]).most_common(1)[0][0]) if features["port_nums"] else 0
        feature_matrix[int(node)] = torch.FloatTensor([out_degree, in_degree, connect_freq, connected_freq, most_freq_port_num])                

    # TODO: Sparse Matrix가 입력으로 들어오면 이걸로... (지금 안됨)
    # Convert scipy sparse matrix -> torch sparse matrix
    # feature_matrix = feature_matrix.tocoo().astype(np.float32)
    # indices = torch.from_numpy(np.vstack((feature_matrix.row, feature_matrix.col)).astype(np.int64))
    # values = torch.from_numpy(feature_matrix.data)
    # shape = torch.Size(feature_matrix.shape)
    # feature_matrix = torch.sparse.FloatTensor(indices, values, shape)
    
    return feature_matrix


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.stack(labels, dim=0)
    return batched_graph, batched_labels


def plot_values(train_values, val_values, title):
    x = list(range(1, len(train_values)+1))
    plt.figure()
    plt.title(title)
    plt.plot(x, train_values, marker='o', label='Training')
    plt.plot(x, val_values, marker='x', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend()
    plt.savefig(title + '.png')
