import torch

from typing import Optional, Sequence, List, Union
from tqdm import tqdm

from torch import nn
from torch.optim import Optimizer

from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    TrainGeneratorAfterExpPlugin,
    EvaluationPlugin,
)
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.logging import InteractiveLogger

from src.continual_learning.plugins import EfficientGenerativeReplayPlugin, TrainDiffusionGeneratorAfterExpPlugin


class GenerativeReplayWithDiffusion(SupervisedTemplate):
    """Generative Replay Strategy

    This implements Deep Generative Replay for a Scholar consisting of a Solver
    and Generator as described in https://arxiv.org/abs/1705.08690.

    The model parameter should contain the solver. As an optional input
    a generator can be wrapped in a trainable strategy
    and passed to the generator_strategy parameter. By default a simple VAE will
    be used as generator.

    For the case where the Generator is the model itself that is to be trained,
    please simply add the GenerativeReplayPlugin() when instantiating
    your Generator's strategy.

    See GenerativeReplayPlugin for more details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        generator_strategy: BaseTemplate = None,
        replay_size: int = None,
        increasing_replay_size: bool = False,
        **base_kwargs
    ):
        """
        Creates an instance of Generative Replay Strategy
        for a solver-generator pair.

        :param model: The solver model.
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
        :param generator_strategy: A trainable strategy with a generative model,
            which employs GenerativeReplayPlugin. Defaults to None.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        # Check if user inputs a generator model
        # (which is wrapped in a strategy that can be trained and
        # uses the GenerativeReplayPlugin;
        # see 'VAETraining" as an example below.)
        if generator_strategy is not None:
            self.generator_strategy = generator_strategy
        else:
            raise ValueError("Please provide a generator strategy.")

        rp = EfficientGenerativeReplayPlugin(
            generator_strategy=self.generator_strategy,
            replay_size=replay_size,
            increasing_replay_size=increasing_replay_size,
        )

        tgp = TrainDiffusionGeneratorAfterExpPlugin()

        if plugins is None:
            plugins = [tgp, rp]
        else:
            plugins.append(tgp)
            plugins.append(rp)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

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
        for self.mbatch in tqdm(self.dataloader):
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