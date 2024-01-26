import os
import torch

from tqdm import tqdm
from copy import deepcopy
from typing import Optional, Tuple
from diffusers import DDIMPipeline, SchedulerMixin, EMAModel

from src.standard_training.losses.diffusion_losses import DiffusionLoss
from src.standard_training.evaluators.base_evaluator import BaseEvaluator
from src.standard_training.trackers.wandb_tracker import WandbTracker
from src.standard_training.trackers.base_tracker import Stage


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
                 save_path: str = "./results/diffusion",
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
        self.save_path = save_path
        self.best_model_path = os.path.join(self.save_path, "best_model")
        self.last_model_path = os.path.join(self.save_path, "last_model")

        # adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
        # alpha = 1.0 - args.model_ema_decay
        # alpha = min(1.0, alpha * adjust)
        # self.model_ema = EMAModel(model.parameters(), power=3/4)

        self.current_epoch = 0
        self.best_auc = torch.inf
        self.best_model = None

        try:
            self.load(self.best_model_path)
        except:
            pass

        os.makedirs(self.best_model_path, exist_ok=True)
        os.makedirs(self.last_model_path, exist_ok=True)

    def load(self, path: str):
        if not os.path.exists(path):
            return

        pipeline = DDIMPipeline.from_pretrained(path)
        self.model = self.model.to("cpu")
        self.model.load_state_dict(pipeline.unet.state_dict())
        self.model = self.model.to(self.device)
        del pipeline

        # Load optimizer and training state
        checkpoint = torch.load(os.path.join(path, "training_state.pt"))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint["current_epoch"]
        self.best_auc = checkpoint["best_auc"]

    def save(self, path: str):
        pipeline = DDIMPipeline(self.model, self.scheduler)
        pipeline.save_pretrained(path)

        # Save optimizer and training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "best_auc": self.best_auc,
        }, os.path.join(path, "training_state.pt"))

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def train(self, train_loader, eval_loader, save_every: int = 1):        
        for epoch in range(self.current_epoch, self.train_epochs):
            print(f"Epoch {epoch}")
            self.current_epoch = epoch

            self.model.train()
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
                # self.model_ema.step()

                if self.tracker is not None:
                    self.tracker.add_batch_metric("loss", loss.item(), step + epoch * len(train_loader))

                average_loss += loss.item()
                bar.set_postfix(loss=average_loss / (step + 1))

            if self.tracker is not None:
                self.tracker.add_epoch_metric("loss", average_loss / len(train_loader), epoch)
                self.tracker.flush()

            auc = torch.inf

            if save_every > 0 and epoch % save_every == 0 and epoch > 0 or epoch == self.train_epochs - 1:
                if self.evaluator is not None:
                    if self.tracker is not None:
                            self.tracker.set_stage(Stage.TEST)

                    self.model.eval()
                    metrics = self.evaluator.evaluate(self.model, eval_loader, epoch, compute_auc=True)
                    auc = metrics["auc"]

                    if self.tracker is not None:
                        for key, value in metrics.items():
                            self.tracker.add_epoch_metric(key, value, epoch)
                        self.tracker.flush()

                if auc <= self.best_auc:
                    print(f"New best model with AUC {auc}")
                    self.best_auc = auc 
                    self.best_model = deepcopy(self.model)
                    self.save(self.best_model_path)
                    
                self.save(self.last_model_path)

        if self.tracker is not None:
            self.tracker.finish()
                
