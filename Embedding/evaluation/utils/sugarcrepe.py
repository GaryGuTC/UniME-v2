import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import Dataset

from src.model_utils import PHI3V, vlm_image_tokens
from src.dataset import process_image

from evaluation.utils.utils import dataset_2_HFdataset_4_sugarcrepe
from evaluation.utils.sugar_crepe import surgar_crepe_dataset_4_LLM

class EvalDataset4sugarcrepe(Dataset):
    def __init__(self, modality, model_backbone, data_args):
        self.model_backbone = model_backbone
        self.data_args = data_args
        self.modality = modality
        self.raw_data = surgar_crepe_dataset_4_LLM(data_args.subset_name)
        self.raw_data = dataset_2_HFdataset_4_sugarcrepe(self.raw_data)

        if modality == "image":
            self.eval_data = self.get_image_data()
        elif modality == "text_pos":
            self.eval_data = self.get_text_pos_data()
        elif modality == "text_neg":
            self.eval_data = self.get_text_neg_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        text, image = self.eval_data[idx]
        if self.model_backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone])
            if self.data_args.image_resolution:
                image = process_image(image, self.data_args.image_resolution)
        return text, image

    def get_image_data(self):
        eval_data = []

        inst = "<|image_1|> Find an image caption describing the given image."
        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
        return eval_data

    def get_text_pos_data(self):
        eval_data = []
        inst = ""
        
        for row in self.raw_data:
            eval_data.append((inst + row["pos_text"], None))
        return eval_data
    
    def get_text_neg_data(self):
        eval_data = []
        inst = ""

        for row in self.raw_data:
            eval_data.append((inst + row["neg_text"], None))
        return eval_data
    

class EvalDataset4sugarcrepe_rerank(Dataset):  
    def __init__(self, model_backbone, data_args):
        self.model_backbone = model_backbone
        self.data_args = data_args
        self.raw_data = surgar_crepe_dataset_4_LLM(data_args.subset_name)
        self.raw_data = dataset_2_HFdataset_4_sugarcrepe(self.raw_data)
        self.eval_data = self.get_eval_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        text, image, candidate_text, candidate_image = self.eval_data[idx]
        if self.model_backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone])
            if self.data_args.image_resolution:
                image = process_image(image, self.data_args.image_resolution)
        return text, image, candidate_text, candidate_image

    def get_eval_data(self):
        eval_data = []
        # i2t
        inst = "<|image_1|> Find an image caption describing the given image."
        inst_text = ""

        for row in self.raw_data:
            eval_data.append((inst, row["image"], inst_text + row["pos_text"], None))
            eval_data.append((inst, row["image"], inst_text + row["neg_text"], None))
        return eval_data