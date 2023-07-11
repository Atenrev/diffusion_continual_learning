import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Optional, Tuple
from diffusers import DDIMPipeline, SchedulerMixin, EMAModel

from src.losses.diffusion_losses import DiffusionLoss
from src.evaluators.base_evaluator import BaseEvaluator
from src.trackers.wandb_tracker import WandbTracker
from src.trackers.base_tracker import Stage


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
                 tracker: Optional[WandbTracker] = None,
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
        self.tracker = tracker

        # adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
        # alpha = 1.0 - args.model_ema_decay
        # alpha = min(1.0, alpha * adjust)
        self.model_ema = EMAModel(model.parameters())

        self.best_model = None


    def save(self, path):
        pipeline = DDIMPipeline(self.model, self.scheduler)
        pipeline.save_pretrained(path)

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def train(self, train_loader, eval_loader, save_path: str = "./results/diffusion", save_every: int = 1):
        best_fid = torch.inf
        
        for epoch in range(self.train_epochs):
            print(f"Epoch {epoch}")

            bar = tqdm(enumerate(train_loader),
                       desc="Training loop", total=len(train_loader))
            average_loss = 0
            
            if self.tracker is not None:
                self.tracker.set_stage(Stage.TRAIN)

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
                self.model_ema.step()

                if self.tracker is not None:
                    self.tracker.add_batch_metric("loss", loss.item(), step + epoch * len(train_loader))

                average_loss += loss.item()
                bar.set_postfix(loss=average_loss / (step + 1))

            if self.tracker is not None:
                self.tracker.add_epoch_metric("loss", average_loss / len(train_loader), epoch)

            fid = torch.inf

            if self.evaluator is not None and (save_every > 0 and epoch % save_every == 0 and epoch > 0 or epoch == self.train_epochs - 1):
                fid = self.evaluator.evaluate(self.model, eval_loader, epoch)["fid"]

                if self.tracker is not None:
                    self.tracker.set_stage(Stage.TEST)
                    self.tracker.add_epoch_metric("fid", fid, epoch)

            if fid <= best_fid:
                best_fid = fid 
                self.best_model = self.model
                self.save(save_path)
                
            self.save(save_path)

        if self.tracker is not None:
            self.tracker.finish()
                
