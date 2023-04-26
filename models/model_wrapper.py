import os
import torch
import torch.nn.functional as F
from models.ppgn_type import PPGN_LGVR_plus
from models.gin_type import GIN_LGVR

class ModelWrapper(object):
    def __init__(self, config, data, pvr):
        self.config = config
        self.data = data

        if self.config.gin:
            self.config.node_labels = self.data.train_graphs[0].node_features.shape[1]

        if self.config.model == 'gin_lgvr':
            self.model = GIN_LGVR(self.config, pvr, False).cuda()
        elif self.config.model == 'gin_lgvr_plus':
            self.model = GIN_LGVR(self.config, pvr, True).cuda()
        else:
            self.model = PPGN_LGVR_plus(self.config, pvr).cuda()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, best: bool, epoch: int, optimizer: torch.optim.Optimizer):
        filename = 'best.tar' if best else 'last.tar'
        print("Saving model as {}...".format(filename), end=' ')
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(self.config.checkpoint_dir, filename))
        print("Model saved.")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, best: bool):
        """
        :param best: boolean, to load best model or last checkpoint
        :return: tuple of optimizer_state_dict, epoch
        """
        filename = 'best.tar' if best else 'last.tar'
        print("Loading {}...".format(filename), end=' ')
        checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(torch.device('cuda'))
        print("Model loaded.")

        return checkpoint['optimizer_state_dict'], checkpoint['epoch']

    def loss_and_results(self, inputs, preds, labels):
        """
        :param inputs: shape N x input_depth x num_nodes x num_nodes
        :param preds: shape N x C if not self.pd and shape [N x C, N x num_nodes x num_nodes] else
        :param labels: shape Nx1 for classification, shape NxC for regression (QM9)
        :return: tuple of (loss tensor, dists numpy array) for QM9
                          (loss tensor, number of correct predictions) for classification graphs
        """
        scores = preds[0]
        pred_adjs = preds[1]
        if self.config.gin:
            true_adjs = torch.cat([g.adj_mat.unsqueeze(0) for g in inputs], 0).to(pred_adjs.device)
        else:
            true_adjs = inputs[:, 0, :, :].squeeze(1)

        if self.config.dataset_name == 'QM9':
            differences = (scores-labels).abs().sum(dim=0)
            loss = differences.sum()
            loss += F.mse_loss(pred_adjs, true_adjs)
            dists = differences.detach().cpu().numpy()
            return loss, dists
        else:
            loss = F.cross_entropy(scores, labels, reduction='sum')
            loss += F.mse_loss(pred_adjs, true_adjs)
            correct_predictions = torch.eq(torch.argmax(scores, dim=1), labels).sum().cpu().item()
            return loss, correct_predictions

    def run_model_get_loss_and_results(self, inputs, labels):
        return self.loss_and_results(inputs, self.model(inputs), labels)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
