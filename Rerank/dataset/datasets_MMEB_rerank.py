import os
import json
from torch.utils.data import Dataset
import random 
random.seed(42)
import json
from PIL import Image
import torch.distributed as dist

DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000
class LazySupervisedDataset(Dataset):

    def __init__(
        self, 
        training_data_path,
        image_path,
        tokenizer = None,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.training_data = json.load(open(training_data_path))
        random.shuffle(self.training_data)
        self.image_path = image_path
        self.tokenizer = tokenizer 
        self.hard_neg_num = len(self.training_data[0]["hard_negatives"])

    def __len__(self) -> int:
        return len(self.training_data)

    def construct_rerank_messages_single_candidate(self, query_dict, cand_dict, type='pos'):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query and a candidate. Please evaluate whether the candidate meets the requirements of the query. If it does, respond with 'Yes'; if it doesn't, responed with 'No'."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Yes" if type == 'pos' else "No"}
                ]
            }
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidate:'}]

        if 'image' in query_dict:
            query.append({'type': 'image', 'image': query_dict['image']})
        if 'txt' in query_dict:
            query.append({'type': 'text', 'text': query_dict['txt']})
        if 'image' in cand_dict:
            cand.append({'type': 'image', 'image': cand_dict['image']})
        if 'txt' in cand_dict:
            cand.append({'type': 'text', 'text': cand_dict['txt']})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

    def construct_rerank_messages_multi_candidates(self, query_dict, cand_lists, ans):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query followed by multiple candidates in the format: (1) cand1 (2) cand2, etc. Each candidate is independent of the others. Evaluate each candidate against the query, and respond with the number corresponding to the candidate that best meets the requirements of the query."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Ans: ({ans})"}
                ]
            }
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidates:'}]

        if 'image' in query_dict:
            query.append({'type': 'image', 'image': query_dict['image']})
        if 'txt' in query_dict:
            query.append({'type': 'text', 'text': query_dict['txt']})
        for i, cand_dict in enumerate(cand_lists):
            cand.append({'type': 'text', 'text': f'({i + 1}) '})
            if 'image' in cand_dict:
                cand.append({'type': 'image', 'image': cand_dict['image']})
            if 'txt' in cand_dict:
                cand.append({'type': 'text', 'text': cand_dict['txt']})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message
        
    def get_instance(self, index, num_candidates):
        final_instance = {}
        ### Select instance
        instance = self.training_data[index]
        ### choose query
        final_instance["query"] = {"text": instance["query_text"]}
        if instance["query_image"] != "": final_instance["query"]["image"] = Image.open(os.path.join(self.image_path, instance["query_image"]))
        ### choose positive
        final_instance['pos_cand'] = {"text": instance["pos_text"]}
        if instance["pos_image"] != "": final_instance["pos_cand"]["image"] = Image.open(os.path.join(self.image_path, instance["pos_image"]))
        ### random select hard negative for pair, now: the top1 in hard negative list
        hard_neg_sample = instance["hard_negatives"][0]
        final_instance['neg_cand'] = {"text": hard_neg_sample[0]}
        if hard_neg_sample[1] != "": final_instance["neg_cand"]["image"] = Image.open(os.path.join(self.image_path, hard_neg_sample[1]))
        ### hard negative list
        neg_cand_lists = []
        for each in instance["hard_negatives"][:5]:
            temp_hard_cand = {}
            temp_hard_cand["text"] = each[0]
            if each[1] != "": temp_hard_cand["image"] = Image.open(os.path.join(self.image_path, each[1]))
            neg_cand_lists.append(temp_hard_cand)
        final_instance['neg_cand_lists'] = neg_cand_lists
        return final_instance 

    def __getitem__(self, i):
        num_candidates = random.randint(1, 4)
        instance = self.get_instance(i, num_candidates)
        query_dict = instance['query']
        cand_dict = instance['pos_cand']
        neg_dict = instance['neg_cand']
        cand_dict_lists = instance['neg_cand_lists']
        rerank_pos_message = self.construct_rerank_messages_single_candidate(query_dict, cand_dict, type='pos')
        rerank_neg_message = self.construct_rerank_messages_single_candidate(query_dict, neg_dict, type='neg')
        # generate random answer position
        ans = random.randint(1, num_candidates + 1)
        cand_dict_lists.insert(ans - 1, cand_dict)
        rerank_multi_candidates_message = self.construct_rerank_messages_multi_candidates(query_dict, cand_dict_lists, ans)
        return rerank_pos_message, rerank_neg_message, rerank_multi_candidates_message