import torch
import torch.nn as nn
import torch.nn.functional as F

class AdjointLoss(nn.Module):
    def __init__(self, alpha=1, training=True):
        super().__init__()
        self.alpha = alpha
        self.training = training

    def forward(self, output, target):
        l = output.shape[0]
        if self.training:
            loss = torch.sum(-target *
                             F.log_softmax(output[:l//2], dim=-1), dim=-1)
            loss = loss.mean()
        else:
            log_preds1 = F.log_softmax(output[:l//2], dim=-1)
            loss = F.nll_loss(log_preds1, target)

        prob1 = F.softmax(output[:l//2], dim=-1)
        prob2 = F.softmax(output[l//2:], dim=-1)
        kl = (prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)

        if not self.training:
            return loss, kl.mean(), loss + self.alpha * 10.0 * kl.mean()
        return loss + self.alpha * 10.0 * kl.mean()