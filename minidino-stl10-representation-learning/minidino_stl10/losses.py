import torch
from torch import nn
from torch.nn import functional as F


class DINOLoss(nn.Module):
    """DINO self-distillation loss with centering and sharpening."""

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        num_teacher_crops: int = 2,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.num_teacher_crops = num_teacher_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        num_student_crops: int,
    ) -> torch.Tensor:
        student_out = (student_output / self.student_temp).chunk(num_student_crops)
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.num_teacher_crops)

        total_loss = 0.0
        n_loss_terms = 0

        for teacher_idx, teacher_prob in enumerate(teacher_out):
            for student_idx, student_logits in enumerate(student_out):
                if student_idx == teacher_idx:
                    continue
                loss = torch.sum(
                    -teacher_prob * F.log_softmax(student_logits, dim=-1),
                    dim=-1,
                )
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss = total_loss / max(1, n_loss_terms)
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)
