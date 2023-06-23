import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Optional, Tuple
from diffusers import DDIMPipeline, SchedulerMixin

from src.losses.diffusion_losses import DiffusionLoss
from src.evaluators.base_evaluator import BaseEvaluator


class DiffusionTraining:

    def __init__(self,
                 model: torch.nn.Module,
                 scheduler: SchedulerMixin,
                 optimizer: torch.optim.Optimizer,
                 criterion: DiffusionLoss,
                 train_mb_size: int,
                 train_epochs: int,
                 eval_mb_size: int,
                 device: str,
                 train_timesteps: int,
                 evaluator: Optional[BaseEvaluator] = None,
                 ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.eval_mb_size = eval_mb_size
        self.device = device
        self.evaluator = evaluator
        self.train_timesteps = train_timesteps

        self.best_model = None

    def save(self, path):
        pipeline = DDIMPipeline(self.model, self.scheduler)
        pipeline.save_pretrained(path)

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def train(self, train_loader, eval_loader, save_path: str = "./results/diffusion"):
        for epoch in range(self.train_epochs):
            print(f"Epoch {epoch}")

            bar = tqdm(enumerate(train_loader),
                       desc="Training loop", total=len(train_loader))
            best_fid = torch.inf

            for step, clean_images in bar:
                self.optimizer.zero_grad()

                batch_size = clean_images["pixel_values"].shape[0]
                clean_images = clean_images["pixel_values"].to(self.device)

                noise = torch.randn(clean_images.shape).to(clean_images.device)
                timesteps = torch.randint(
                    0, self.train_timesteps, (batch_size,), device=self.device
                ).long()
                noisy_images = self.scheduler.add_noise(
                    clean_images, noise, timesteps)

                noise_pred = self.model(
                    noisy_images, timesteps, return_dict=False)[0]

                loss = self.criterion(noise_pred, noise, timesteps)

                loss.backward()
                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                bar.set_postfix(loss=loss.item())

            fid = torch.inf

            if self.evaluator is not None:
                fid = self.evaluator.evaluate(self.model, eval_loader, epoch)["fid"]

            if fid <= best_fid:
                best_fid = fid 
                self.best_model = self.model
                self.save(save_path)
                
            self.save(save_path)
                
