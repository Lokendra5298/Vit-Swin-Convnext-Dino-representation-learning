import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError('features must have shape [batch, views, dim]')
        device = features.device
        batch_size, n_views, dim = features.shape
        features = F.normalize(features, dim=-1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_features = contrast_features
        anchor_count = n_views

        logits = torch.div(torch.matmul(anchor_features, contrast_features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, n_views)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        positives_per_anchor = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(positives_per_anchor, min=1.0)
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
