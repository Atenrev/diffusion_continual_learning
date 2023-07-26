import os
import numpy as np
import torch
import wandb

from typing import List, Tuple

from src.trackers.base_tracker import Stage, ExperimentTracker


class WandbTracker(ExperimentTracker):
    """
    Creates a tracker that implements the ExperimentTracker protocol and logs to wandb.
    """

    def __init__(self, configs: dict, experiment_name: str, project_name: str, tags: List[str] = None):
        self.stage = Stage.TRAIN

        self.run = wandb.init(project=project_name,
                              name=experiment_name, 
                              config=configs,
                              tags=tags)

        wandb.define_metric("batch_step")
        wandb.define_metric("epoch")

        # TODO: Make metrics dynamic 
        for metric in ["loss"]: 
            for stage in Stage:
                wandb.define_metric(f"{stage.name}/batch_{metric}", step_metric='batch_step')
                wandb.define_metric(f"{stage.name}/epoch_{metric}", step_metric='epoch')

        wandb.define_metric(f"{Stage.TEST}/batch_fid", step_metric='batch_step')
        wandb.define_metric(f"{Stage.TEST}/epoch_fid", step_metric='epoch')

    def set_stage(self, stage: Stage):
        self.stage = stage

    def add_batch_metric(self, name: str, value: float, step: int, commit: bool = True):
        wandb.log({f"{self.stage.name}/batch_{name}": value, "batch_step": step}, commit=commit)

    def add_epoch_metric(self, name: str, value: float, step: int, commit: bool = True):
        wandb.log({f"{self.stage.name}/epoch_{name}": value, "epoch": step}, commit=commit)
        
    def flush(self):
        pass

    def finish(self):
        wandb.finish()