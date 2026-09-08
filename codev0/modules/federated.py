import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from data.loader import DataLoader


class ServerModule:
    def __init__(self, args, sd, gpu_server):
        self.args = args
        self.gpu_id = gpu_server
        self.sd = sd

    def aggregate(self, local_weights, ratio=None):
        aggregated = {key: None for key in local_weights[0].keys()}
        if ratio is None:
            ratio = [1.0 / len(local_weights)] * len(local_weights)
        for name in aggregated:
            aggregated[name] = np.sum([weights[name] * ratio[idx] for idx, weights in enumerate(local_weights)], axis=0)
        return aggregated


class ClientModule:
    def __init__(self, args, w_id, g_id, sd):
        self.sd = sd
        self.gpu_id = g_id
        self.worker_id = w_id
        self.args = args
        self.loader = DataLoader(self.args)

    def switch_state(self, client_id):
        self.client_id = client_id
        self.loader.switch(client_id)
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return os.path.exists(os.path.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))

    def init_state(self):
        raise NotImplementedError()

    def save_state(self):
        raise NotImplementedError()

    def load_state(self):
        raise NotImplementedError()

    def validation_step(self, batch, mask=None):
        self.model.eval()
        y_hat = self.model(batch)
        if torch.sum(mask).item() == 0:
            return y_hat, 0.0
        if self.args.dataset in ['Minesweeper', 'Tolokers', 'Questions']:
            loss = F.binary_cross_entropy_with_logits(y_hat[mask].view(-1), batch.y[mask].view(-1))
        else:
            loss = F.cross_entropy(y_hat[mask], batch.y[mask])
        return y_hat, loss.item()

    def accuracy(self, preds, targets):
        if targets.size(0) == 0:
            return 1.0
        if self.args.dataset in ['Minesweeper', 'Tolokers', 'Questions']:
            targets_np = targets.cpu().detach().numpy().reshape(-1)
            preds_np = preds.cpu().detach().numpy().reshape(-1)
            if np.unique(targets_np).size < 2:
                return 0.5
            return roc_auc_score(targets_np, preds_np)
        pred_labels = preds.max(1)[1]
        return pred_labels.eq(targets).sum().item() / targets.size(0)

    def f1(self, preds, targets):
        if targets.size(0) == 0:
            return 1.0
        targets_np = targets.long().cpu().detach().numpy()
        if self.args.dataset in ['Minesweeper', 'Tolokers', 'Questions']:
            pred_labels = (preds.view(-1) > 0).long()
            average = 'binary'
        else:
            pred_labels = preds.max(1)[1]
            average = 'macro'
        return f1_score(targets_np, pred_labels.cpu().detach().numpy(), average=average, zero_division=0)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
