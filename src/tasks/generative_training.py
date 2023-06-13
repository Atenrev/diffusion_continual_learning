import os
import torch

from tqdm import tqdm
from typing import Optional, Any

from src.evaluators.base_evaluator import BaseEvaluator


class GenerativeTraining:

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

    def save(self, path: str, epoch: int):
        model_path = os.path.join(path, "model")
        os.makedirs(model_path, exist_ok=True)

        # Save model and scheduler
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, os.path.join(model_path, f"model_{epoch}.pt"))

    def train(self, train_loader, eval_loader, save_path: str = "./results/generative"):
        if self.evaluator is not None:
            assert eval_loader is not None

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

            if self.evaluator is not None:
                self.evaluator.on_epoch_end(self.model, eval_loader, epoch, save_path=save_path)

            self.save(save_path, epoch)
                
