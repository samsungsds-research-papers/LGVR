import torch
import torch.nn as nn
import layers.layers as layers
import layers.modules as modules
from torchph.nn import SLayerRationalHat

class BaseModel(nn.Module):
    def __init__(self, config, ph):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.ph = ph

        self.config = config
        use_new_suffix = config.architecture.new_suffix  # True or False
        block_features = config.architecture.block_features  # List of number of features in each regular block
        original_features_num = config.node_labels + 1  # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(config, last_layer_features, next_layer_features)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # Persistence Diagram induced by PPGN
        self.ph_1_mlp = nn.Conv1d(in_channels=block_features[-1], out_channels=int(0.5*block_features[-1]), kernel_size=1)
        self.ph_1_act = nn.ReLU()
        self.ph_2_mlp = nn.Conv1d(in_channels=int(0.5*block_features[-1]), out_channels=1, kernel_size=1)
        self.ph_2_act = nn.Sigmoid()

        # SLayerRationalHat version
        # Persistence Diagram -> SLayerRationalHat
        self.slrh_0 = SLayerRationalHat(n_elements=int(0.25*block_features[-1]), point_dimension=2, radius_init=0.1)
        self.slrh_0_ess = SLayerRationalHat(n_elements=int(0.125*block_features[-1]), point_dimension=1, radius_init=0.1)
        self.slrh_1_ess = SLayerRationalHat(n_elements=int(0.125 * block_features[-1]), point_dimension=1, radius_init=0.1)

        # # PPGN Vectex Features -> Transformed Vertex Features (for the combination of Persistence Feature Vectors)
        # self.vert1_conv = nn.Conv2d(in_channels=block_features[-1], out_channels=2*block_features[-1], kernel_size=1)
        # self.vert1_act_conv = nn.ReLU()
        # self.vert2_conv = nn.Conv2d(in_channels=2*block_features[-1], out_channels=2*block_features[-1], kernel_size=1)
        # self.vert2_act_conv = nn.ReLU()

        # Second part
        self.fc_layers = nn.ModuleList()
        self.ph_fc_layers = nn.ModuleList()
        if use_new_suffix:
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = modules.FullyConnected(2*output_features, self.config.num_classes, activation_fn=None)
                self.fc_layers.append(fc)
                ph_fc_layers = nn.Linear(in_features=int(0.5 * block_features[-1]), out_features=block_features[-1])
                self.ph_fc_layers.append(ph_fc_layers)
                self.ph_fc_layers.append(nn.ReLU())
                ph_fc_layers = nn.Linear(in_features=block_features[-1], out_features=2 * block_features[-1])
                self.ph_fc_layers.append(ph_fc_layers)

        else:  # use old suffix
            # Sequential fc layers
            self.fc_layers.append(modules.FullyConnected(2*block_features[-1], 512))
            self.fc_layers.append(modules.FullyConnected(512, 256))
            self.fc_layers.append(modules.FullyConnected(256, self.config.num_classes, activation_fn=None))

        self.idx = 0

    def forward(self, input):
        x = input
        scores = torch.tensor(0, device=input.device, dtype=x.dtype)
        edge_idx = [(adj[0] > 0).nonzero().t() for adj in input]

        for i, block in enumerate(self.reg_blocks):

            x = block(x)

            if self.config.architecture.new_suffix:
                # use new suffix
                v_feats = torch.diagonal(x, dim1=-2, dim2=-1)
                v_feats = self.ph_1_act(self.ph_1_mlp(v_feats))
                v_feats = self.ph_2_act(self.ph_2_mlp(v_feats))

                ph_input = []
                for num, k in enumerate(range(x.shape[0])):
                    v_feats_t = torch.squeeze(v_feats[k]).cuda()
                    e = get_boundary_info(edge_idx[k])
                    ph_input.append((v_feats_t, [e]))
                pers = self.ph(ph_input)

                h_0_sub = [t[0][0] for t in pers]
                h_0_ess_sub = [t[1][0].unsqueeze(1) for t in pers]
                h_1_ess_sub = [t[1][1].unsqueeze(1) for t in pers]

                pers_feat = torch.cat(
                    [self.slrh_0(h_0_sub), self.slrh_0_ess(h_0_ess_sub), self.slrh_1_ess(h_1_ess_sub)], dim=1)

                pers_feat = self.ph_fc_layers[3*i+1]((self.ph_fc_layers[3*i](pers_feat)))
                pers_feat = self.ph_fc_layers[3*i+2](pers_feat)

                scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x) + pers_feat) + scores
                # for ablation study
                # scores = self.fc_layers[i](pers_feat) + scores
                # scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x)) + scores

        if not self.config.architecture.new_suffix:
            # old suffix

            # x.shape = (batch_size, block_features[-1] (=400), num_vertices, num_vertices)
            # v_feats.shape = (batch_size, block_features[-1] (=400), num_vertices)
            v_feats = torch.diagonal(x, dim1=-2, dim2=-1)

            # v_feats.shape = (batch_size, 1, num_vertices)
            v_feats = self.ph_1_act(self.ph_1_mlp(v_feats))
            v_feats = self.ph_2_act(self.ph_2_mlp(v_feats))

            ph_input = []
            for num, i in enumerate(range(x.shape[0])):
                v_feats_t = torch.squeeze(v_feats[i])
                e = get_boundary_info(edge_idx[i])
                ph_input.append((v_feats_t.cuda(), [e]))
            pers = self.ph(ph_input)

            h_0_sub = [t[0][0] for t in pers]
            h_0_ess_sub = [t[1][0].unsqueeze(1) for t in pers]
            h_1_ess_sub = [t[1][1].unsqueeze(1) for t in pers]

            pers_feat = torch.cat(
                [self.slrh_0(h_0_sub), self.slrh_0_ess(h_0_ess_sub), self.slrh_1_ess(h_1_ess_sub)], dim=1)

            pers_feat = self.ph_act1_layers(self.ph_fc1_layers(pers_feat))
            pers_feat = self.ph_act2_layers(self.ph_fc2_layers(pers_feat))

            # N: batch size, F: feature dimension, M: the number of vertices
            # diag_offdiag_maxpool version
            x = layers.diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
            x = x + pers_feat
            for fc in self.fc_layers:
                x = fc(x)
            scores = x

        return scores


def get_boundary_info(edge_index):
    e = edge_index.permute(1, 0).sort(1)[0].tolist()
    e = set([tuple(ee) for ee in e])
    return torch.tensor([ee for ee in e], dtype=torch.long).cuda()
