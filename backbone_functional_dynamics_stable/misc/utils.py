import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch


def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)
    tmp_path = fpath + '.tmp'
    torch.save(data, tmp_path)
    os.replace(tmp_path, fpath)


def torch_load(base_dir, filename):
    return torch.load(os.path.join(base_dir, filename), map_location=torch.device('cpu'), weights_only=False)


def estimate_train_homophily(edge_index, labels, train_mask, default=0.75):
    if edge_index is None or edge_index.numel() == 0:
        return float(default)
    train_mask = train_mask.bool()
    row, col = edge_index
    valid_edges = train_mask[row] & train_mask[col]
    if valid_edges.sum().item() == 0:
        return float(default)
    labels = labels.view(-1)
    same_label = labels[row[valid_edges]] == labels[col[valid_edges]]
    return float(same_label.float().mean().item())


def seed_everything(seed, torch_seed=True, cuda=False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch_seed:
        torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, filename), 'w') as outfile:
        json.dump(data, outfile)


def get_state_dict(model):
    state_dict = OrderedDict()
    for key, value in model.state_dict().items():
        state_dict[key] = value.clone().detach().cpu().numpy()
    return state_dict


def set_state_dict(model, state_dict, gpu_id, skip_stat=False):
    tensor_state = OrderedDict()
    current_state = model.state_dict()
    for key, value in state_dict.items():
        if skip_stat and ('running' in key or 'tracked' in key):
            tensor_state[key] = current_state[key]
        elif len(np.shape(value)) == 0:
            tensor_state[key] = torch.tensor(value).cuda(gpu_id)
        else:
            tensor_state[key] = torch.tensor(value).requires_grad_().cuda(gpu_id)
    model.load_state_dict(tensor_state, strict=False)
