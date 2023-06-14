import os
import numpy as np
import torch
import wandb

from pathlib import Path
from typing import List, Tuple

from src.trackers.base_tracker import Stage, ExperimentTracker
from src.common.utils import create_experiment_dir


class WandbTracker(ExperimentTracker):
    """
    Creates a tracker that implements the ExperimentTracker protocol and logs to wandb.
    """

    def __init__(self, log_path: str, configs: dict, experiment_name: str, project_name: str, tags: List[str] = None):
        self.stage = Stage.TRAIN

        self.run = wandb.init(project=project_name,
                              name=experiment_name, 
                              config=configs,
                              tags=tags)
        log_dir, self.models_dir = create_experiment_dir(
            root=log_path, experiment_name=experiment_name)
        self._validate_log_dir(log_dir, create=True)

        wandb.define_metric("batch_step")
        wandb.define_metric("epoch")

        # TODO: Make metrics dynamic 
        for metric in ["loss"]: 
            for stage in Stage:
                wandb.define_metric(f"{stage.name}/batch_{metric}", step_metric='batch_step')
                wandb.define_metric(f"{stage.name}/epoch_{metric}", step_metric='epoch')

        wandb.define_metric(f"{Stage.TEST}/batch_fid", step_metric='batch_step')
        wandb.define_metric(f"{Stage.TEST}/epoch_fid", step_metric='epoch')
        
    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def set_stage(self, stage: Stage):
        self.stage = stage

    def add_batch_metric(self, name: str, value: float, step: int, commit: bool = True):
        wandb.log({f"{self.stage.name}/batch_{name}": value, "batch_step": step}, commit=commit)

    def add_epoch_metric(self, name: str, value: float, step: int):
        wandb.log({f"{self.stage.name}/epoch_{name}": value, "epoch": step})

    @staticmethod
    def collapse_batches(
        y_true: List[np.ndarray], y_pred: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.concatenate(y_true), np.concatenate(y_pred)
        
    def flush(self):
        pass

    def finish(self):
        wandb.finish()