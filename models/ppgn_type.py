import torch
import torch.nn as nn
import layers.layers as layers
import layers.modules as modules

class PPGN_LGVR_plus(nn.Module):
    def __init__(self, config, pvr):
        """
            config: configs for GIN_LGVR
            pvr: pershom.pershom_backend.__C.VRCompCuda__vr_persistence
        """
        super().__init__()

        self.pvr = pvr

        self.config = config
        block_features = config.architecture.block_features  # List of number of features in each regular block
        original_features_num = config.node_labels + 1  # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(config, last_layer_features, next_layer_features)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # LGVR component
        self.fc_layers = nn.ModuleList()

        self.extra_eps = nn.Parameter(torch.zeros(3, dtype=torch.float))

        self.sp_edge_layers = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        self.metric_layers = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.ph_layers = nn.ModuleList()
        self.ph_0_init_layers = nn.ModuleList()
        self.ph_1_init_layers = nn.ModuleList()
        self.ph_0_layers = nn.ModuleList()
        self.ph_1_layers = nn.ModuleList()

        self.feat_dim = 2 * block_features[-1]
        self.last_block_feature = block_features[-1]
        self.ph_init_dim = 16
        ph_st_num_heads = 4
        ph_st_num_inds = 50
        ph_st_ln = True
        self.ph_0_dim = 200
        self.ph_1_dim = 200

        self.sp_edge_layers.append(nn.Conv1d(in_channels=block_features[-1], out_channels=1, kernel_size=1))
        self.edge_layers.append(nn.Conv2d(in_channels=self.feat_dim, out_channels=block_features[-1], kernel_size=1))

        self.metric_layers.append(nn.Conv2d(in_channels=self.feat_dim, out_channels=self.feat_dim, kernel_size=1))
        self.metric_layers.append(nn.Conv2d(in_channels=self.feat_dim, out_channels=block_features[-1], kernel_size=1))
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
        self.fc_layers.append(modules.FullyConnected(self.feat_dim, 512, dropout=self.config.dropout))
        self.fc_layers.append(modules.FullyConnected(512, 256, dropout=self.config.dropout))
        self.fc_layers.append(modules.FullyConnected(256, self.config.num_classes, activation_fn=None, dropout=self.config.dropout))

    def forward(self, input):
        x = input
        total_batch_pred_adjs = torch.zeros((input.size(0), input.size(-1), input.size(-1))).to(input.device)
        # adj_exp.size = (batch_size, 2*block_features[-1] (=800), num_nodes, num_nodes)
        adj_exp = input[:, 0, :, :].unsqueeze(1).expand((input.size(0), self.feat_dim, input.size(-1), input.size(-1))).to(input.device)

        for i, block in enumerate(self.reg_blocks):
            # block(x) : N x input_depth x m x m (N: batch size, m: num_nodes, input_depth: node feature dim)
            x = block(x)

        batch_pers_feat = torch.empty(0, self.feat_dim).to(input.device)
        # v_feats.size = (batch_size, block_features[-1] (=400), num_nodes)
        v_feats = torch.diagonal(x, dim1=-2, dim2=-1)
        # sp_edge_feats.size = (batch_size, 1, num_nodes)
        sp_edge_feats = self.sp_edge_layers[0](v_feats)
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
        edge_feats = self.sigmoid(self.metric_layers[2](self.relu(self.metric_layers[1](edge_feats))))

        for edge_feat in edge_feats:
            init_ph_0_feat = torch.empty(0).to(input.device)
            init_ph_1_feat = torch.empty(0).to(input.device)
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

        x = layers.diag_offdiag_maxpool(x) + (1. + self.extra_eps[2]) * batch_pers_feat

        for fc in self.fc_layers:
            x = fc(x)
        scores = x

        return [scores, total_batch_pred_adjs]
