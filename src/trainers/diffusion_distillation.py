import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Optional, Any, Tuple
from diffusers import DDIMPipeline

from src.losses.diffusion_losses import DiffusionLoss
from diffusers import SchedulerMixin

from src.evaluators.base_evaluator import BaseEvaluator
from src.trainers.base_trainer import BaseTrainer
from src.trackers.wandb_tracker import WandbTracker, Stage


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
                 tracker: Optional[WandbTracker] = None,
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

    def save(self, path):
        pipeline = DDIMPipeline(self.model, self.scheduler)
        pipeline.save_pretrained(path)

    def forward(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def train(self, teacher: torch.nn.Module, eval_loader: Optional[Any] = None, save_every: int = 100, save_path: str = "./results/diffusion_distillation"):
        if self.evaluator is not None:
            assert eval_loader is not None

        if self.tracker is not None:
            self.tracker.set_stage(Stage.TRAIN)

        self.teacher = teacher
        bar = tqdm(enumerate(range(self.train_iterations)), desc="Training loop", total=self.train_iterations)
        best_fid = torch.inf
        average_loss = 0
        n_steps = 0

        for step, iteration in bar:
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

            if (iteration + 1) % save_every == 0 or iteration == self.train_iterations - 1:
                fid = torch.inf

                if self.evaluator is not None:
                    fid = self.evaluator.evaluate(self.model, eval_loader, iteration)["fid"]

                if fid <= best_fid:
                    best_fid = fid 
                    self.best_model = self.model # flawed lol
                    self.save(save_path)

                if self.tracker is not None:
                    self.tracker.add_epoch_metric("loss", average_loss / n_steps, step // save_every)
                    self.tracker.set_stage(Stage.TEST)
                    self.tracker.add_epoch_metric("fid", fid, step // save_every)
                    self.tracker.set_stage(Stage.TRAIN)

                average_loss = 0
                n_steps = 0

        if self.tracker is not None:
            self.tracker.finish()

        return best_fid


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