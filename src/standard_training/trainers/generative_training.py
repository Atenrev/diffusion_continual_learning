import os
import torch

from tqdm import tqdm
from typing import Optional, Any

from src.standard_training.evaluators.base_evaluator import BaseEvaluator
from src.standard_training.trainers.base_trainer import BaseTrainer


class GenerativeTraining(BaseTrainer):

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
        super().__init__(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device, evaluator)

    def train(self, train_loader, eval_loader, save_path: str = "./results/generative"):
        if self.evaluator is not None:
            assert eval_loader is not None

        best_fid = torch.inf

        for epoch in range(self.train_epochs):
            bar = tqdm(train_loader, desc=f"Training epoch {epoch}", total=len(train_loader))

            for batch in bar:
                self.optimizer.zero_grad()

                batch = batch["pixel_values"].to(self.device)
                pred = self.model(batch)
                loss = self.criterion(batch, pred)

                loss.backward()
                self.optimizer.step()

                bar.set_postfix(loss=loss.item())

            fid = torch.inf

            if self.evaluator is not None:
                fid = self.evaluator.evaluate(self.model, eval_loader, epoch)["fid"]

            if fid <= best_fid:
                best_fid = fid 
                self.best_model = self.model
                self.save(save_path, epoch)
                
