import torch
from torch import nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss.

    Args:
        temperature: softmax temperature.
        base_temperature: scaling temperature used in the original SupCon formulation.

    Input:
        features: tensor of shape [batch_size, num_views, feature_dim]
        labels: tensor of shape [batch_size]
    """

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch_size, num_views, feature_dim]")

        device = features.device
        batch_size, num_views, feature_dim = features.shape

        features = nn.functional.normalize(features, dim=-1)
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_features = contrast_features
        anchor_count = num_views

        logits = torch.div(torch.matmul(anchor_features, contrast_features.T), self.temperature)

        # Numerical stability.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, num_views)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(positive_count, min=1.0)

        loss = -self.temperature / self.base_temperature * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
