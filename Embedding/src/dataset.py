import os
import json
import pickle
import random
from PIL import Image, ImageFile
from typing import List

import datasets
from datasets import load_dataset

from torch.jit import isinstance
from torch.utils.data import Dataset

from src.model_utils import PHI3V, vlm_image_tokens

random.seed(42)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "mid_336": # for llavaOV training
        image = image.resize((336, 336))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

class UniME_v2_dataset(Dataset):
    def __init__(self, data_args, model_args):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone
        if data_args.hard_negaitve_data_path.endswith(".json"):
            self.train_data = json.load(open(data_args.hard_negaitve_data_path))
        else:
            self.train_data = pickle.load(open(data_args.hard_negaitve_data_path, 'rb'))
        random.shuffle(self.train_data)
        self.select_hard_negative_num = min(len(self.train_data[0]["hard_negatives"]), data_args.select_hard_negative_num)
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, data_idx):
        train_data = self.get_rerank_data(self.train_data[data_idx])

        output = []
        for each in train_data:
            qry_texts, qry_image_paths, pos_texts, pos_image_paths, score0, score1 = (each["qry"], each["qry_image_path"], each["pos_text"], each["pos_image_path"], each["score0"], each["score1"])
            neg_texts, neg_image_paths = '', None
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            neg_texts = [neg_texts]
            neg_image_paths = [neg_image_paths]
            score0 = [score0]
            score1 = [score1]
            _qry_texts, _qry_images, _pos_texts, _pos_images, _neg_texts, _neg_images, _score0s, _score1s = [], [], [], [], [], [], [], []
            backbone = self.model_args.model_backbone
            for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path, score0, score1 \
                in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths, neg_texts, neg_image_paths, score0, score1):
                if backbone != PHI3V:
                    qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                    pos_text = pos_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone])
                    neg_text = neg_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[backbone]) if neg_text else None
                qry_image = self._get_image(qry_image_path)
                pos_image = self._get_image(pos_image_path)
                neg_image = self._get_image(neg_image_path) if neg_image_path else None
                
                if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                    print("empty inputs")
                    continue
                _qry_texts.append(qry_text)
                _qry_images.append(qry_image)
                _pos_texts.append(pos_text)
                _pos_images.append(pos_image)
                _neg_texts.append(neg_text)
                _neg_images.append(neg_image)
                _score0s.append(score0)
                _score1s.append(score1)
    
            output.append({"query_text": _qry_texts, 
                           "query_image": _qry_images,
                            "pos_text": _pos_texts, 
                            "pos_image": _pos_images,
                            "neg_text": _neg_texts,
                            "neg_image": _neg_images,
                            "score0": _score0s, 
                            "score1": _score1s})
        return output

    def _get_image(self, img_path):
        if img_path == "": return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def get_rerank_data(self, row):
        tmp_data = [{"qry" : row["query_text"], 
                        "qry_image_path": row["query_image"], 
                        "pos_text": row["pos_text"], 
                        "pos_image_path": row["pos_image"],
                        "score0": 0.0,
                        "score1": row["query_pos_scores"]}]
        for tmpi in range(0, self.select_hard_negative_num, 2):
            tmp_data += [{"qry" : row["hard_negatives"][tmpi][0], 
                            "qry_image_path": row["hard_negatives"][tmpi][1], 
                            "pos_text": row["hard_negatives"][tmpi+1][0], 
                            "pos_image_path": row["hard_negatives"][tmpi+1][1],
                            "score0": row["hard_negatives_scores"][tmpi],
                            "score1": row["hard_negatives_scores"][tmpi+1]}]
        return tmp_data

class EvalDataset_rerank(Dataset):
    def __init__(self, data_args, model_args, subset):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone

        self.eval_data = json.load(open(os.path.join(data_args.encode_output_path, f"{subset}_rerank_topk.json")))
        self.candidate_num_4_listwise = 10

        self.rerank_data = self.eval_data
        self.reranked_dataset = datasets.Dataset.from_dict({
            "qry_text": [pair["qry_text"] for pair in self.rerank_data],
            "qry_img_path": [pair["qry_img_path"] for pair in self.rerank_data],
            "cand_data": [pair["candidates_topk"] for pair in self.rerank_data]
        })


    def __len__(self):
        return len(self.rerank_data)

    def __getitem__(self, item):
        qry_text, qry_img_path, cand_data = \
            self.reranked_dataset[item]["qry_text"], self.reranked_dataset[item]["qry_img_path"], \
                self.reranked_dataset[item]["cand_data"]
        if self.backbone != PHI3V:
            qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone])
            new_cand_data = []
            for cand_text, cand_image_path in cand_data:
                new_cand_data.append([cand_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone]), self._get_image(cand_image_path)])
        return qry_text, self._get_image(qry_img_path), new_cand_data

    def _get_image(self, img_path):
        if img_path == "": return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def get_rerank_data(self):
        total_data = []
        for row in self.eval_data:
            total_data += [{"qry_text" : row["qry_text"], "qry_img_path": row["qry_img_path"], "cand_text": each_cand[0], "cand_img_path": each_cand[1]}  for each_cand in row["candidates_topk"]]
        return total_data


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone

        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.backbone])

        return text, self._get_image(img_path),

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data