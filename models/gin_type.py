import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
import layers.modules as modules


class GIN_LGVR(nn.Module):
    def __init__(self, config, pvr, plus):
        '''
            config: configs for GIN_LGVR
            pvr: pershom.pershom_backend.__C.VRCompCuda__vr_persistence
            plus: whether the model is GIN_LGVR_plus or not
        '''

        super().__init__()

        self.pvr = pvr
        self.plus = plus
        self.num_layers = len(config.architecture.block_features)
        self.final_dropout = config.dropout
        self.learn_eps = True
        self.neighbor_pooling_type = 'sum'
        self.graph_pooling_type = 'sum'
        self.device = 'cuda'
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        num_mlp_layers = config.architecture.depth_of_mlp
        input_dim = config.node_labels
        hidden_dim = config.architecture.block_features[0]
        block_features = config.architecture.block_features

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # LGVR component
        self.fc_layers = nn.ModuleList()
        self.extra_eps = nn.Parameter(torch.zeros(3, dtype=torch.float))

        self.sp_edge_layers = nn.ModuleList()
        self.metric_layers = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.ph_layers = nn.ModuleList()
        self.ph_0_init_layers = nn.ModuleList()
        self.ph_1_init_layers = nn.ModuleList()
        self.ph_0_layers = nn.ModuleList()
        self.ph_1_layers = nn.ModuleList()

        self.feat_dim = block_features[-1]
        self.last_block_feature = block_features[-1]
        self.ph_init_dim = 8
        ph_st_num_heads = 4
        ph_st_num_inds = 50
        ph_st_ln = True
        self.ph_0_dim = 64
        self.ph_1_dim = 64

        self.sp_edge_layers.append(nn.Conv1d(in_channels=block_features[-1], out_channels=1, kernel_size=1))
        self.metric_layers.append(nn.Conv2d(in_channels=2 * self.feat_dim, out_channels=2 * self.feat_dim, kernel_size=1))
        self.metric_layers.append(nn.Conv2d(in_channels=2 * self.feat_dim, out_channels=block_features[-1], kernel_size=1))
        self.metric_layers.append(nn.Conv2d(in_channels=block_features[-1], out_channels=1, kernel_size=1))

        self.ph_0_init_layers.append(modules.FullyConnected(2, self.ph_init_dim))
        self.ph_0_init_layers.append(modules.FullyConnected(self.ph_init_dim, self.ph_init_dim))
        self.ph_1_init_layers.append(modules.FullyConnected(2, self.ph_init_dim))
        self.ph_1_init_layers.append(modules.FullyConnected(self.ph_init_dim, self.ph_init_dim))

        self.ph_0_st_enc = nn.Sequential(
            modules.ISAB(self.ph_init_dim, self.ph_init_dim, ph_st_num_heads, ph_st_num_inds, ln=ph_st_ln),
            modules.ISAB(self.ph_init_dim, self.ph_init_dim, ph_st_num_heads, ph_st_num_inds, ln=ph_st_ln),
            modules.ISAB(self.ph_init_dim, self.ph_init_dim, ph_st_num_heads, ph_st_num_inds, ln=ph_st_ln))
        self.ph_0_st_dec = nn.Sequential(
            modules.PMA(self.ph_init_dim, ph_st_num_heads, 1, ln=ph_st_ln),
            modules.FullyConnected(self.ph_init_dim, self.ph_init_dim, activation_fn=None))

        self.ph_1_st_enc = nn.Sequential(
            modules.ISAB(self.ph_init_dim, self.ph_init_dim, ph_st_num_heads, ph_st_num_inds, ln=ph_st_ln),
            modules.ISAB(self.ph_init_dim, self.ph_init_dim, ph_st_num_heads, ph_st_num_inds, ln=ph_st_ln),
            modules.ISAB(self.ph_init_dim, self.ph_init_dim, ph_st_num_heads, ph_st_num_inds, ln=ph_st_ln))
        self.ph_1_st_dec = nn.Sequential(
            modules.PMA(self.ph_init_dim, ph_st_num_heads, 1, ln=ph_st_ln),
            modules.FullyConnected(self.ph_init_dim, self.ph_init_dim, activation_fn=None))

        self.ph_0_layers.append(modules.FullyConnected(2 * self.ph_init_dim, 4 * self.ph_init_dim))
        self.ph_0_layers.append(modules.FullyConnected(4 * self.ph_init_dim, self.ph_0_dim))
        self.ph_1_layers.append(modules.FullyConnected(2 * self.ph_init_dim, 4 * self.ph_init_dim))
        self.ph_1_layers.append(modules.FullyConnected(4 * self.ph_init_dim, self.ph_1_dim))

        self.ph_layers.append(modules.FullyConnected(self.ph_0_dim + self.ph_1_dim, self.feat_dim))
        self.ph_layers.append(modules.FullyConnected(self.feat_dim, self.feat_dim, activation_fn=None))

        # Sequential fc layers
        self.fc_layers.append(modules.FullyConnected(self.feat_dim, 64, dropout=config.dropout))
        self.fc_layers.append(modules.FullyConnected(64, 32, dropout=config.dropout))
        self.fc_layers.append(modules.FullyConnected(32, config.num_classes, activation_fn=None, dropout=config.dropout))

    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        # compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                # add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is assumed to be stored in -1
                pad.extend([-1] * (max_deg - len(pad)))

                # Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

    def split_to_batch(self, batch_graph, h):
        g_len = len(batch_graph[0].g)
        for i, graph in enumerate(batch_graph):
            if g_len != len(graph.g):
                raise Exception("incorrect graph size!!!")

        output = h.reshape((len(batch_graph), g_len, self.feat_dim))

        return output

    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1. / len(graph.g)] * len(graph.g))

            else:
                ###sum pooling
                elem.extend([1] * len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        # list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)
            hidden_rep.append(h)

        pooled_h = torch.spmm(graph_pool, h)

        batch_pers_feat = torch.empty(0, self.feat_dim).to(self.device)
        adj_exp = torch.cat([graph.adj_mat.unsqueeze(0) for graph in batch_graph], 0)
        # adj_exp.size = (batch_size, block_features[-1], num_nodes, num_nodes)
        adj_exp = adj_exp.unsqueeze(1).expand(
            (adj_exp.size(0), 2 * self.feat_dim, adj_exp.size(-1), adj_exp.size(-1))).to(self.device)
        total_batch_pred_adjs = torch.zeros((adj_exp.size(0), adj_exp.size(-1), adj_exp.size(-1))).to(self.device)

        input_ph = self.split_to_batch(batch_graph, h)
        # v_feats.size = (batch_size, block_features[-1], num_nodes)
        v_feats = input_ph.transpose(-1, -2)
        # sp_edge_feats.size = (batch_size, 1, num_nodes)
        sp_edge_feats = self.sp_edge_layers[0](v_feats)
        # v_feats.size = (batch_size, block_features[-1], num_nodes)
        v_feats = v_feats + (1. + self.extra_eps[0]) * sp_edge_feats

        # for line graph computation
        # edge_feats.size = (batch_size, 2*block_features[-1], num_nodes, num_nodes)
        edge_feats = torch.cat((v_feats.unsqueeze(-1) + v_feats.unsqueeze(-2),
                                torch.abs(v_feats.unsqueeze(-1) - v_feats.unsqueeze(-2))),
                               dim=1)
        edge_feats = self.relu(self.metric_layers[0](edge_feats))
        edge_feats *= adj_exp  # for message passing of line graph
        edge_feats = (1. + self.extra_eps[1] - 2.) * edge_feats + \
                     (torch.sum(edge_feats, dim=-1).unsqueeze(-1) + torch.sum(edge_feats, dim=-2).unsqueeze(-2))
        edge_feats = self.sigmoid(self.metric_layers[2](
            self.relu(self.metric_layers[1](edge_feats))))

        for edge_feat in edge_feats:
            init_ph_0_feat = torch.empty(0).to(self.device)
            init_ph_1_feat = torch.empty(0).to(self.device)
            for feat in edge_feat:
                vr_pd = self.pvr(1. - feat, 1, 0.5)

                ess_0 = vr_pd[1][0].reshape((-1, 1))
                ess_1 = vr_pd[1][1].reshape((-1, 1))
                ref_ess_0 = torch.cat((ess_0, torch.ones_like(ess_0)), -1)
                ref_ess_1 = torch.cat((ess_1, torch.ones_like(ess_1)), -1)
                h_0 = torch.cat((vr_pd[0][0], ref_ess_0), 0)
                h_1 = torch.cat((vr_pd[0][1], ref_ess_1), 0)

                init_h_0 = self.ph_0_init_layers[1](self.ph_0_init_layers[0](h_0))
                init_h_1 = self.ph_1_init_layers[1](self.ph_1_init_layers[0](h_1))
                init_h_0_sum = torch.sum(init_h_0, dim=0)
                init_h_1_sum = torch.sum(init_h_1, dim=0)

                st_0_feat = self.ph_0_st_dec(self.ph_0_st_enc(init_h_0.unsqueeze(0)))
                st_1_feat = self.ph_1_st_dec(self.ph_1_st_enc(init_h_1.unsqueeze(0)))
                init_ph_0_feat = torch.cat([init_ph_0_feat, torch.cat([init_h_0_sum.squeeze(), st_0_feat.squeeze()])])
                init_ph_1_feat = torch.cat([init_ph_1_feat, torch.cat([init_h_1_sum.squeeze(), st_1_feat.squeeze()])])

            h_0_feat = self.ph_0_layers[1](self.ph_0_layers[0](init_ph_0_feat.unsqueeze(0)))
            h_1_feat = self.ph_1_layers[1](self.ph_1_layers[0](init_ph_1_feat.unsqueeze(0)))
            total_pd = torch.cat([h_0_feat, h_1_feat], dim=1)

            pers_feat = self.ph_layers[1](self.ph_layers[0](total_pd))
            batch_pers_feat = torch.cat([batch_pers_feat, pers_feat], dim=0)

        total_batch_pred_adjs += torch.sum(edge_feats, dim=1)
        if self.plus:
            x = pooled_h + (1. + self.extra_eps[2]) * batch_pers_feat
        else:
            x = batch_pers_feat

        for fc in self.fc_layers:
            x = fc(x)
        scores = x

        return [scores, total_batch_pred_adjs]
