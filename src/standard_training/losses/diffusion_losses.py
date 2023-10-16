import torch

from abc import ABC
from typing import Optional
from torch.nn import functional as F
from diffusers import SchedulerMixin

from src.common.utils import extract_into_tensor


class DiffusionLoss(ABC):

    def __init__(self, scheduler: SchedulerMixin):
        self.scheduler = scheduler

    def __call__(self, target: torch.Tensor, pred: torch.Tensor, timesteps: Optional[torch.Tensor] = None):
        raise NotImplementedError


class MSELoss(DiffusionLoss):

    def __init__(self, scheduler: SchedulerMixin):
        super().__init__(scheduler)

    def __call__(self, target: torch.Tensor, pred: torch.Tensor, timesteps: Optional[torch.Tensor] = None):
        loss = F.mse_loss(target, pred)
        return loss
    

class SmoothL1Loss(DiffusionLoss):
    
        def __init__(self, scheduler: SchedulerMixin):
            super().__init__(scheduler)
    
        def __call__(self, target: torch.Tensor, pred: torch.Tensor, timesteps: Optional[torch.Tensor] = None):
            loss = F.smooth_l1_loss(target, pred)
            return loss


class MinSNRLoss(DiffusionLoss):
    """
    Based on https://github.com/TiankaiHang/Min-SNR-Diffusion-Training
    """

    def __init__(self, scheduler: SchedulerMixin, k: int = 5, divide_by_snr: bool = True, reduction: str = "mean"):
        super().__init__(scheduler)
        self.k = k
        self.divide_by_snr = divide_by_snr
        self.reduction = reduction

    def __call__(self, target: torch.Tensor, pred: torch.Tensor, timesteps: Optional[torch.Tensor] = None):
        assert timesteps is not None

        sqrt_alphas_cumprod = (self.scheduler.alphas_cumprod ** 0.5)
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod) ** 0.5
        alpha = extract_into_tensor(
            sqrt_alphas_cumprod, timesteps, timesteps.shape)
        sigma = extract_into_tensor(
            sqrt_one_minus_alpha_prod, timesteps, timesteps.shape)
        snr = (alpha / sigma) ** 2
        mse_loss_weight = torch.stack(
            [snr, self.k * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        
        if self.divide_by_snr:
            mse_loss_weight = mse_loss_weight / snr
            
        loss = mse_loss_weight * F.mse_loss(target, pred)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise NotImplementedError

        return loss
