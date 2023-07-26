import os
import torch

from tqdm import tqdm
from copy import deepcopy
from typing import Optional, Any, Tuple
from diffusers import DDIMPipeline

from src.losses.diffusion_losses import DiffusionLoss
from diffusers import SchedulerMixin

from src.evaluators.base_evaluator import BaseEvaluator
from src.trainers.base_trainer import BaseTrainer
from src.trackers.wandb_tracker import ExperimentTracker, Stage


class DiffusionDistillation(BaseTrainer):

    def __init__(self,
                 model: torch.nn.Module,
                 scheduler: SchedulerMixin,
                 optimizer: torch.optim.Optimizer,
                 criterion: DiffusionLoss,
                 train_mb_size: int,
                 train_iterations: int,
                 eval_mb_size: int,
                 device: str,
                 train_timesteps: int,
                 evaluator: Optional[BaseEvaluator] = None,
                 tracker: Optional[ExperimentTracker] = None,
                 *args, **kwargs
                 ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_mb_size = train_mb_size
        self.train_iterations = train_iterations
        self.eval_mb_size = eval_mb_size
        self.device = device
        self.evaluator = evaluator
        self.train_timesteps = train_timesteps
        self.tracker = tracker
        self.best_model = None

    def save(self, path, iteration):
        # Save model
        pipeline = DDIMPipeline(self.model, self.scheduler)
        pipeline.save_pretrained(path)

        # Save optimizer and training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "current_iteration": iteration,
        }, os.path.join(path, "training_state.pt"))

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def train(self, teacher: torch.nn.Module, eval_loader: Optional[Any] = None, save_every: int = 100, save_path: str = "./results/diffusion_distillation", start_iteration: int = 0):
        if self.evaluator is not None:
            assert eval_loader is not None

        if self.tracker is not None:
            self.tracker.set_stage(Stage.TRAIN)

        best_model_path = os.path.join(save_path, "best_model")
        last_model_path = os.path.join(save_path, "last_model")
        os.makedirs(best_model_path, exist_ok=True)
        os.makedirs(last_model_path, exist_ok=True)

        self.teacher = teacher
        bar = tqdm(range(start_iteration, self.train_iterations), desc="Training loop", total=self.train_iterations)
        best_metrics = {"auc": torch.inf}
        average_loss = 0
        n_steps = 0

        for step in bar:
            self.optimizer.zero_grad()

            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (self.train_mb_size,), device=self.device
            ).long()

            pred, target = self.forward(timesteps)

            loss = self.criterion(target, pred, timesteps)

            loss.backward()
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            bar.set_postfix(loss=loss.item())
            average_loss += loss.item()
            n_steps += 1

            if self.tracker is not None:
                self.tracker.add_batch_metric("loss", loss.item(), step)

            if (step + 1) % save_every == 0 or step == self.train_iterations - 1:
                epoch = step // save_every
                auc = torch.inf

                if self.evaluator is not None:
                    metrics = self.evaluator.evaluate(self.model, eval_loader, step, compute_auc=True)
                    auc = metrics["auc"]

                if self.tracker is not None and self.evaluator is not None:
                    self.tracker.add_epoch_metric("loss", average_loss / n_steps, epoch)
                    self.tracker.set_stage(Stage.TEST)
                    
                    for key, value in metrics.items():
                        self.tracker.add_epoch_metric(key, value, epoch)

                    self.tracker.flush()
                    self.tracker.set_stage(Stage.TRAIN)

                if auc <= best_metrics["auc"]:
                    best_metrics = metrics 
                    self.best_model = deepcopy(self.model)
                    self.save(best_model_path, step)

                average_loss = 0
                n_steps = 0
                self.save(last_model_path, step)

        if self.tracker is not None:
            self.tracker.finish()

        return best_metrics


class GaussianDistillation(DiffusionDistillation):

    def __init__(self,
                generation_steps: int = 20,
                eta: float = 0.0,
                *args, **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.generation_steps = generation_steps
        self.eta = eta

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = self.model.config.in_channels
        sample_size = self.model.config.sample_size
        noise = torch.randn((self.train_mb_size, channels, sample_size, sample_size)).to(self.device)

        target = self.teacher(noise, timesteps, return_dict=False)[0]
        student_pred = self.model(noise, timesteps, return_dict=False)[0]

        return student_pred, target


class GaussianSymmetryDistillation(DiffusionDistillation):

    def __init__(self,
                generation_steps: int = 20,
                eta: float = 0.0,
                *args, **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.generation_steps = generation_steps
        self.eta = eta

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            timesteps: tensor of shape (batch_size, ) containing the timesteps to use for the diffusion process
        """
        channels = self.model.config.in_channels
        sample_size = self.model.config.sample_size
        noise = torch.randn((self.train_mb_size, channels, sample_size, sample_size)).to(self.device)

        # Symmetrize noise
        # From timesteps 1000 to 666, we do not symmetrize the noise
        # From timesteps 666 to 333, we symmetrize the noise vertically or horizontally
        # From timesteps 333 to 1, we symmetrize the noise vertically and horizontally
        t_666_333_idx = torch.where((timesteps <= 666) & (timesteps > 333))[0]
        # Half of the noise is symmetrized vertically, the other half horizontally 
        t_666_333_idx_h = t_666_333_idx[:len(t_666_333_idx)//2]
        t_666_333_idx_v = t_666_333_idx[len(t_666_333_idx)//2:]
        t_333_1_idx = torch.where(timesteps <= 333)[0]
        # Mirror horizontally
        noise[t_666_333_idx_h, :, :sample_size//2, :] = torch.flip(noise[t_666_333_idx_h, :, sample_size//2:, :], dims=[2,])
        # Mirror vertically
        noise[t_666_333_idx_v, :, :, :sample_size//2] = torch.flip(noise[t_666_333_idx_v, :, :, sample_size//2:], dims=[3,])
        # Mirror horizontally and vertically
        noise[t_333_1_idx, :, :sample_size//2, :] = torch.flip(noise[t_333_1_idx, :, sample_size//2:, :], dims=[2,])
        noise[t_333_1_idx, :, :, :sample_size//2] = torch.flip(noise[t_333_1_idx, :, :, sample_size//2:], dims=[3,])

        target = self.teacher(noise, timesteps, return_dict=False)[0]
        student_pred = self.model(noise, timesteps, return_dict=False)[0]

        return student_pred, target
    

class GenerationDistillation(DiffusionDistillation):

    def __init__(self,
                generation_steps: int = 20,
                eta: float = 0.0,
                *args, **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.generation_steps = generation_steps
        self.eta = eta

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = self.model.config.in_channels
        sample_size = self.model.config.sample_size
        noise = torch.randn((self.train_mb_size, channels, sample_size, sample_size)).to(self.device)

        generated_images = self.teacher.generate(self.train_mb_size)
        noisy_images = self.scheduler.add_noise(generated_images, noise, timesteps)
        with torch.no_grad():
            target = self.teacher(noisy_images, timesteps, return_dict=False)[0]
        student_pred = self.model(noisy_images, timesteps, return_dict=False)[0]

        return student_pred, target

            
class PartialGenerationDistillation(DiffusionDistillation):

    def __init__(self,
                generation_steps: int = 20,
                eta: float = 0.0,
                *args, **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.generation_steps = generation_steps
        self.eta = eta

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        generated_images = self.teacher.generate(self.train_mb_size, timesteps)
        with torch.no_grad():
            target = self.teacher(generated_images, timesteps, return_dict=False)[0]
        student_pred = self.model(generated_images, timesteps, return_dict=False)[0]
        return student_pred, target
    

class NoDistillation(DiffusionDistillation):
    
        def __init__(self,
                    generation_steps: int = 20,
                    eta: float = 0.0,
                    *args, **kwargs
                    ):
            super().__init__(*args, **kwargs)
            self.generation_steps = generation_steps
            self.eta = eta
    
        def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            channels = self.model.config.in_channels
            sample_size = self.model.config.sample_size
            noise = torch.randn((self.train_mb_size, channels, sample_size, sample_size)).to(self.device)
            target = noise

            generated_images = self.teacher.generate(self.train_mb_size)
            noisy_images = self.scheduler.add_noise(generated_images, noise, timesteps)
            student_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
    
            return student_pred, target