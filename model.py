import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 graph_pooling_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for _ in range(n_layers-1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # final aggregating layer
        self.layers.append(SAGEConv(n_hidden, out_feats, aggregator_type, feat_drop=dropout, activation=activation))

        # final pooling
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        
        # final mapping layer
        self.final = nn.Linear(out_feats, n_classes)
    
    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        pool_emb = self.pool(g, h)
        final_emb = self.final(pool_emb)
        return final_emb
