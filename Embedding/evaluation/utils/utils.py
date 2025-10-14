import os
import json
import math
import pickle
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from datasets import Dataset as HFDataset

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True

def dataset_2_HFdataset(dataset):
    images = []
    captions = []
    for i in range(len(dataset)):
        image, caption = dataset[i]
        images.append(image) 
        captions.append(caption)
    data_dict = {
        'image': images,
        'text': captions
    }
    dataset = HFDataset.from_dict(data_dict)
    return dataset

def recall_at_k(scores, 
                positive_pairs, 
                k):
    nb_texts, nb_images = scores.shape # bs, bs
    topk_indices = torch.topk(scores, k, dim=1)[1] # 
    nb_positive = positive_pairs.sum(dim=1) 
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def recall_k_candiadates(scores, 
                         positive_pairs, 
                         k):
    topk_indices = torch.topk(scores, k, dim=1)[1]
    return topk_indices

def batchify(func, 
             X, 
             Y, 
             batch_size, 
             device, 
             *args, 
             **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

def dataset_2_HFdataset_4_sugarcrepe(dataset):
    images = []
    pos_list = []
    neg_list = []
    for i in range(len(dataset)):
        pos, neg, image = dataset[i]
        images.append(image)
        pos_list.append(pos)
        neg_list.append(neg)
    data_dict = {
        'image': images,
        'pos_text': pos_list,
        'neg_text': neg_list
    }
    dataset = HFDataset.from_dict(data_dict)
    return dataset

def get_pred(qry_t, tgt_t, normalization=False, topk=10):
    """
    Use L2 norms.
    """
    if normalization:
        qry_t_norm = np.linalg.norm(qry_t)
        tgt_t_norms = np.linalg.norm(tgt_t, axis=1)
        scores = np.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        scores = np.dot(tgt_t, qry_t)
    topk = min(topk, len(scores))
    preds = np.argpartition(scores, -topk)[-topk:]
    preds = preds[np.argsort(-scores[preds])]
    return scores, preds

def save_results(results, model_args, data_args, train_args):
    save_file = model_args.model_name + "_" + (model_args.model_type if  model_args.model_type is not None else "") + "_" + data_args.embedding_type + "_results.json"
    with open(os.path.join(data_args.encode_output_path, save_file), "w") as json_file:
        json.dump(results, json_file, indent=4)

def print_results(results):
    for dataset, acc in results.items():
        print(dataset, ",", acc)

def gather_object(encode_path, features, is_distributed=False, rerank=False):
    if not os.path.exists(encode_path):
        if is_distributed:
            world_size = dist.get_world_size()
            gather_features = [None for _ in range(world_size)]
            dist.all_gather_object(gather_features, features)
            
            if is_rank_zero():
                total_features = []
                for rank_features in gather_features: total_features.extend(rank_features)
                if not rerank: total_features = np.array(total_features)
                
                with open(encode_path, 'wb') as f:
                    pickle.dump(total_features, f)
        else:
            with open(encode_path, 'wb') as f: pickle.dump(features, f)


class SequentialDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        from torch.distributed import get_world_size, get_rank, is_initialized
        
        if num_replicas is None:
            if not is_initialized():
                num_replicas = 1
            else:
                num_replicas = get_world_size()
        if rank is None:
            if not is_initialized():
                rank = 0
            else:
                rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            import torch
            g = torch.Generator()
            g.manual_seed(0)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[-padding_size:]

        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples