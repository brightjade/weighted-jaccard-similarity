import os
import csv
from collections import defaultdict
import numpy as np
import torch
import dgl
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import makedirs, save_info, load_info
from scipy.sparse import lil_matrix
from tqdm import tqdm

from utils import convert_labels_to_ids, update_feature_matrix


class TCPDataset(DGLDataset):
    """[summary]

    Args:
        DGLDataset ([type]): [description]
    """
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False, 
                 verbose=False):
        super(TCPDataset, self).__init__(name='tcp_connection',
                                         url=url,
                                         raw_dir=raw_dir,
                                         save_dir=save_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        """process raw data to graphs, labels, splitting masks"""
        self.graphs = []
        self.labels = []

        if self.raw_dir == './train':    
            for f in tqdm(os.listdir(self.raw_dir), desc='Processing Training Files'):
                src_ids, dst_ids = [], []
                con_types = set()

                # Dictionary Mapping of Node Features {"nodeID_1" : {"in_neighbors": set, ...}, ...}
                node_dict = defaultdict(lambda: {"in_neighbors": set(), "out_neighbors": set(), "connect_freq": 0, "connected_freq": 0, "port_nums": []})

                # Read & store graph information from given file
                with open(os.path.join(self.raw_dir, f), "r") as tsv:
                    tsv_reader = csv.reader(tsv, delimiter='\t')
                    for (src_id, dst_id, port_num, timestamp, con_type) in tsv_reader:
                        node_dict[src_id]["out_neighbors"].add(dst_id)
                        node_dict[src_id]["connect_freq"] += 1
                        node_dict[src_id]["port_nums"].append(port_num)
                        node_dict[dst_id]["in_neighbors"].add(src_id)
                        node_dict[dst_id]["connected_freq"] += 1
                        node_dict[dst_id]["port_nums"].append(port_num)
                        src_ids.append(int(src_id))
                        dst_ids.append(int(dst_id))
                        con_types.add(con_type)
                    label = torch.FloatTensor(convert_labels_to_ids(con_types))
                
                # Create DGL graph
                g = dgl.graph(data=(src_ids, dst_ids), idtype=torch.long)

                # Convert Graph -> Vectors(Matrix)
                # TODO: Sparse Matrix로 구현 (지금 안됨)
                # feature_matrix = lil_matrix((g.num_nodes(), len(node_dict[0])))
                feature_matrix = torch.zeros((g.num_nodes(), len(node_dict[0])))
                feature_matrix = update_feature_matrix(feature_matrix, node_dict)
                g.ndata['attr'] = feature_matrix
                self.graphs.append(g)
                self.labels.append(label)

        elif self.raw_dir == './valid_query':
            for f in tqdm(os.listdir(self.raw_dir), desc='Processing Validation Query Files'):
                src_ids, dst_ids = [], []

                # Dictionary Mapping of Node Features {"nodeID_1" : {"in_neighbors": set, ...}, ...}
                node_dict = defaultdict(lambda: {"in_neighbors": set(), "out_neighbors": set(), "connect_freq": 0, "connected_freq": 0, "port_nums": []})
                
                # Read & store graph information from given file
                with open(os.path.join(self.raw_dir, f), "r") as tsv:
                    tsv_reader = csv.reader(tsv, delimiter='\t')
                    for (src_id, dst_id, port_num, timestamp) in tsv_reader:
                        node_dict[src_id]["out_neighbors"].add(dst_id)
                        node_dict[src_id]["connect_freq"] += 1
                        node_dict[src_id]["port_nums"].append(port_num)
                        node_dict[dst_id]["in_neighbors"].add(src_id)
                        node_dict[dst_id]["connected_freq"] += 1
                        node_dict[dst_id]["port_nums"].append(port_num)
                        src_ids.append(int(src_id))
                        dst_ids.append(int(dst_id))
                
                # Create DGL graph
                g = dgl.graph(data=(src_ids, dst_ids), idtype=torch.long)

                # Convert Graph -> Vectors(Matrix)
                # TODO: Sparse Matrix로 구현 (지금 안됨)
                # feature_matrix = lil_matrix((g.num_nodes(), len(node_dict[0])))
                feature_matrix = torch.zeros((g.num_nodes(), len(node_dict[0])))
                feature_matrix = update_feature_matrix(feature_matrix, node_dict)
                g.ndata['attr'] = feature_matrix
                self.graphs.append(g)

            # Validation gets labels from different files
            for f in tqdm(os.listdir('./valid_answer'), desc='Processing Validation Answer Files'):
                if os.stat(os.path.join('./valid_answer', f)).st_size == 0: # if file is empty, label is 'benign' connection.
                    label = torch.FloatTensor(convert_labels_to_ids([]))
                    self.labels.append(label)
                else:
                    with open(os.path.join('./valid_answer', f), "r") as tsv:
                        tsv_reader = csv.reader(tsv, delimiter='\t')
                        for con_types in tsv_reader:
                            label = torch.FloatTensor(convert_labels_to_ids(set(con_types)))
                            self.labels.append(label)

    def __getitem__(self, idx):
        """ Get graph and label by index

        Args:
            idx (Int): Item index

        Returns:
            Tuple: (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)
        
    # def save(self):
    #     # save processed data to directory `self.save_path`
    #     graph_path = os.path.join(self.save_path, '_dgl_graph.bin')
    #     save_graphs(graph_path, self.graphs, {'labels': self.labels})
    #     info_path = os.path.join(self.save_path, '_info.pkl')
    #     save_info(info_path, {'port_nums': self.port_nums, 'timestamps': self.timestamps})

    # def load(self):
    #     # load processed data from directory `self.save_path`
    #     graph_path = os.path.join(self.save_path, '_dgl_graph.bin')
    #     self.graphs, label_dict = load_graphs(graph_path)
    #     self.labels = label_dict['labels']
    #     info_path = os.path.join(self.save_path, '_info.pkl')
    #     self.port_nums = load_info(info_path)['port_nums']
    #     self.timestamps = load_info(info_path)['timestamps']

    # def has_cache(self):
    #     # check whether there are processed data in `self.save_path`
    #     graph_path = os.path.join(self.save_path, '_dgl_graph.bin')
    #     info_path = os.path.join(self.save_path, '_info.pkl')
    #     return os.path.exists(graph_path) and os.path.exists(info_path)
