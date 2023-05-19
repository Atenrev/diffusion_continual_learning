import torch

from typing import Optional, Sequence, List, Union

from torch import nn
from torch.optim import Optimizer
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.templates import SupervisedTemplate
from avalanche.logging import InteractiveLogger


def get_default_vae_logger():
    return EvaluationPlugin(loggers=[InteractiveLogger()])


class DifussionTraining(SupervisedTemplate):
    """
    Difussion Training class.

    This strategy implements the Difussion Training strategy
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion=nn.SmoothL1Loss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = get_default_vae_logger(),
        eval_every=-1,
        timesteps: int = 300,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=None, # Temporal. TODO: Evaluate difussion model
            eval_every=eval_every,
            **base_kwargs
        )
        self.timesteps = timesteps

    def criterion(self):
        """
        Returns the loss criterion.
        """
        return self._criterion(self.noise_x, self.mb_output)
    
    # def forward(self):
    #     """
    #     Forward pass.
    #     """
    #     return self.model(self.mb_x, self.t_x, self.noise_x)
    
    def training_epoch(self, **kwargs):
        """
        Training epoch.

        :param kwargs:
        :return:
        """        
        for self.mbatch in self.dataloader:
            batch_size = self.mbatch[0].shape[0]

            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)

            # Sample noise
            # self.t_x = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
            # self.noise_x = torch.randn_like(self.mb_x)
            self.mb_output, self.noise_x = self.forward()

            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)