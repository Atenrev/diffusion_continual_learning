import os
import torch

from typing import Any, Optional
from abc import ABC

from src.standard_training.evaluators.base_evaluator import BaseEvaluator


class BaseTrainer(ABC):

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: Any,
                 train_mb_size: int,
                 train_epochs: int,
                 eval_mb_size: int,
                 device: str,
                 evaluator: Optional[BaseEvaluator] = None,
                 ):
        """
        Class for training generative models in a traditional way.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.eval_mb_size = eval_mb_size
        self.device = device
        self.evaluator = evaluator        
        self.best_model = None

    def save(self, path: str, epoch: int):
        model_path = os.path.join(path, "model")
        os.makedirs(model_path, exist_ok=True)

        # Save model and scheduler
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, os.path.join(model_path, f"model_{epoch}.pt"))

    def train(self, train_loader: Any, eval_loader: Any, save_path: str = "./results/generative", save_every: int = 1, **kwargs):
        raise NotImplementedError

    def evaluate(self, eval_loader, save_path: str = "./results/generative"):
        assert self.evaluator is not None
        assert self.best_model is not None
        metrics = self.evaluator.evaluate(self.best_model, eval_loader, 0, save_path=save_path)
        return metrics