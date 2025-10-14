import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist

class SoftLabelSoftKLContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, beta=1.0): 
        super().__init__()
        self.temperature = temperature
        self.beta = beta

    def forward(self, scores, understanding_scores):
        understanding_scores = F.softmax(understanding_scores / self.temperature, dim=-1)
        scores = F.softmax(scores / self.temperature, dim=-1)
        log_understanding = torch.log(understanding_scores + 1e-8)
        log_scores = torch.log(scores + 1e-8)
        # KL(P||Q)
        kl_pq = F.kl_div(log_scores, understanding_scores, reduction='batchmean', log_target=False)
        # KL(Q||P)
        kl_qp = F.kl_div(log_understanding, scores, reduction='batchmean', log_target=False)
        # Symmetric KL loss
        loss = 0.5 * (kl_pq + kl_qp)
        # if dist.get_rank() == 0: print(f" ==== kl_pq: {kl_pq} | kl_qp: {kl_qp} | loss: {loss} ==== ") #! open to check KL loss 
        return loss

class DistributedSoftLabelSoftKLContrastiveLoss(SoftLabelSoftKLContrastiveLoss):
    def __init__(self, scale_loss: bool = True, temperature: float = 0.01, select_hard_negative_num: int=8):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature
        self.select_hard_negative_num = select_hard_negative_num # 8
        self.interval = int((self.select_hard_negative_num/2)+1)

    def __call__(self, 
                 x: Tensor, 
                 y: Tensor, 
                 x_understanding_score: Tensor,
                 y_understanding_score: Tensor,
                 **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        dist_x_understanding_score = self.gather_tensor(x_understanding_score)
        dist_y_understanding_score = self.gather_tensor(y_understanding_score)

        qry = dist_x[::self.interval, :]
        pos = dist_y[::self.interval, :]
        total_hard = []
        for i in range(1, self.interval): 
            total_hard.append(dist_x[i::self.interval, :])
            total_hard.append(dist_y[i::self.interval, :])
        target_score = []
        for i in range(self.interval):
            target_score.append(dist_x_understanding_score[i::self.interval])
            target_score.append(dist_y_understanding_score[i::self.interval])
        target_score = torch.stack(target_score, dim=1)
        understanding_scores = target_score[:,1:]

        qry = torch.unsqueeze(qry, dim=1)
        pos = torch.unsqueeze(pos, dim=1)
        total_hard = torch.stack(total_hard, dim=1)
        x = qry 
        y = torch.concat([pos,total_hard], dim=1) 
        scores = torch.bmm(x, y.transpose(1,2)).squeeze(1)
        loss = super().__call__(scores, understanding_scores, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss
 
    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
