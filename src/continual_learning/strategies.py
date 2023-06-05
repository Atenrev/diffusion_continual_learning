import torch

from typing import Optional, Sequence, List, Union
from tqdm import tqdm

from torch import nn
from torch.optim import Optimizer

from avalanche.models.pnn import PNN
from avalanche.models import VAE_loss
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.logging import InteractiveLogger
from diffusers import SchedulerMixin

from src.continual_learning.plugins import UpdatedGenerativeReplayPlugin, TrainDiffusionGeneratorAfterExpPlugin


class WeightedSoftGenerativeReplay(SupervisedTemplate):
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

    In this implementation, the criterion is a weighted sum of the
    real data loss and the replay data loss. The replay data loss is
    scaled by the temperature parameter T.

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
        T: float = 2.0,
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

        rp = UpdatedGenerativeReplayPlugin(
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

        self.untrained_solver = True
        self.replay_size = replay_size
        self.T = T

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

    def criterion(self):
        """
        Compute the weighted loss for the current minibatch.
        """
        if self.experience.current_experience == 0 or not self.model.training:
            return self._criterion(self.mb_output, self.mb_y)
        
        mb_y_replay = self.mb_y[self.train_mb_size:]
        # Scale logits by temperature according to M. van de Ven et al. (2020)
        mb_y_replay = mb_y_replay / self.T
        mb_y_replay = mb_y_replay.softmax(dim=1)
        real_data_loss = self._criterion(self.mb_output[:self.train_mb_size], self.mb_y[:self.train_mb_size])
        replay_data_loss = self._criterion(self.mb_output[self.train_mb_size:], mb_y_replay)
        replay_data_loss = replay_data_loss * self.T**2
        return ((1/(self.experience.current_experience+1)) * real_data_loss
                + (1 - (1/(self.experience.current_experience+1))) * replay_data_loss)


def get_default_vae_logger():
    return EvaluationPlugin(loggers=[InteractiveLogger()])


class DiffusionTraining(SupervisedTemplate):
    """
    Difussion Training class.

    This strategy implements the Difussion Training strategy
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: SchedulerMixin,
        optimizer: Optimizer,
        criterion=nn.SmoothL1Loss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = get_default_vae_logger(),
        eval_every=-1,
        train_timesteps: int = 1000,
        generation_timesteps: int = 10,
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
        self.scheduler = scheduler
        self.train_timesteps = train_timesteps
        self.generation_timesteps = generation_timesteps
    
    def criterion(self):
        """
        Compute the loss for the current minibatch.
        """
        if self.experience.current_experience == 0 or not self.model.training:
            return self._criterion(self.noise_x, self.mb_output)
        
        real_data_loss = self._criterion(self.noise_x[:self.train_mb_size], self.mb_output[:self.train_mb_size])
        replay_data_loss = self._criterion(self.noise_x[self.train_mb_size:], self.mb_output[self.train_mb_size:])
        return ((1/(self.experience.current_experience+1)) * real_data_loss
                + (1 - (1/(self.experience.current_experience+1))) * replay_data_loss)
    
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
            self.noise_x = torch.randn(self.mbatch[0].shape).to(self.device)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device
            ).long()
            noisy_images = self.scheduler.add_noise(self.mbatch[0], self.noise_x, timesteps)

            self.mb_output = self.model(noisy_images, timesteps, return_dict=False)[0]

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

    def _after_training_exp(self, **kwargs):
        self.untrained_solver = False
        return super()._after_training_exp(**kwargs)
        

class VAETraining(SupervisedTemplate):
    """VAETraining class

    This is the training strategy for the VAE model
    found in the models directory.
    We make use of the SupervisedTemplate, even though technically this is not a
    supervised training. However, this reduces the modification to a minimum.

    We only need to overwrite the criterion function in order to pass all
    necessary variables to the VAE loss function.
    Furthermore we remove all metrics from the evaluator.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion=VAE_loss,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = get_default_vae_logger(),
        eval_every=-1,
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
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

    def criterion(self):
        """
        Adapt input to criterion as needed to compute reconstruction loss
        and KL divergence. See default criterion VAELoss.
        """
        if self.experience.current_experience == 0 or not self.model.training:
            return self._criterion(self.mb_x, self.mb_output)
        
        mb_output_real = (
            self.mb_output[0][:self.train_mb_size],
            self.mb_output[1][:self.train_mb_size],
            self.mb_output[2][:self.train_mb_size],
        )
        mb_output_replay = (
            self.mb_output[0][self.train_mb_size:],
            self.mb_output[1][self.train_mb_size:],
            self.mb_output[2][self.train_mb_size:],
        )
        real_data_loss = self._criterion(self.mb_x[:self.train_mb_size], mb_output_real)
        replay_data_loss = self._criterion(self.mb_x[self.train_mb_size:], mb_output_replay)
        return ((1/(self.experience.current_experience+1)) * real_data_loss
                + (1 - (1/(self.experience.current_experience+1))) * replay_data_loss)