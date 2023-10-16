import torch

from typing import Optional, Iterable, List, Union, Sequence
from copy import deepcopy

from torch import nn
from torch.optim import Optimizer

from avalanche.models import VAE_loss
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    EWCPlugin,
    SynapticIntelligencePlugin,
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.benchmarks import CLExperience, CLStream
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.logging import InteractiveLogger
from diffusers import SchedulerMixin

from src.continual_learning.plugins import UpdatedGenerativeReplayPlugin, TrainGeneratorAfterExpPlugin
from src.continual_learning.metrics.diffusion_metrics import DiffusionMetricsMetric


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

        tgp = TrainGeneratorAfterExpPlugin()

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

        row_sums = torch.sum(self.mb_y, dim=1)
        not_sum_one_mask = (row_sums != 1.0)
        index = torch.nonzero(not_sum_one_mask, as_tuple=False)

        if index.numel() > 0:
            start_of_replay = index[0, 0].item()
        else:
            return self._criterion(self.mb_output, self.mb_y)

        real_data_loss = self._criterion(
            self.mb_output[:start_of_replay], self.mb_y[:start_of_replay])

        mb_y_replay = self.mb_y[start_of_replay:]
        # Scale logits by temperature according to M. van de Ven et al. (2020)
        mb_y_replay = mb_y_replay / self.T
        mb_y_replay = mb_y_replay.log_softmax(dim=1)

        output_replay = self.mb_output[start_of_replay:]
        output_replay = output_replay / self.T
        output_replay = output_replay.softmax(dim=1)

        replay_data_loss = -output_replay * mb_y_replay
        replay_data_loss = replay_data_loss.sum(dim=1).mean()
        replay_data_loss = replay_data_loss * self.T**2

        return ((1/(self.experience.current_experience+1)) * real_data_loss
                + (1 - (1/(self.experience.current_experience+1))) * replay_data_loss)


def get_default_generator_logger():
    return EvaluationPlugin(
        # [DiffusionMetricsMetric()],
        loggers=[InteractiveLogger()]
    )


class BaseDiffusionTraining(SupervisedTemplate):
    """
    Difussion Training class.

    This strategy implements the Difussion Training strategy
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: SchedulerMixin,
        optimizer: Optimizer,
        criterion=nn.MSELoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = get_default_generator_logger(),
        eval_every=-1,
        train_timesteps: int = 1000,
        replay_start_timestep: int = 0,
        generation_timesteps: int = 10,
        lambd: float = 1,
        weight_replay_loss: bool = True,
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
        :param train_timesteps: The number of timesteps to train the model for.
        :param generation_timesteps: The number of timesteps to generate for.
        :param lambd: The lambda parameter for the replay loss.
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
        self.scheduler = scheduler
        self.train_timesteps = train_timesteps
        self.generation_timesteps = generation_timesteps
        self.old_model = None
        self.untrained_generator = True
        self.train_exp_id = 0
        self.replay_start_timestep = replay_start_timestep
        self.lambd = lambd
        self.weight_replay_loss = weight_replay_loss

    def generate_samples(self, batch_size):
        """
        Generate samples from the generator.
        """
        return self.model.generate(batch_size, output_type="torch")

    def criterion(self):
        """
        Compute the loss for the current minibatch.
        """
        if self.experience.current_experience == 0 or not self.model.training:
            return self._criterion(self.noise_x, self.mb_output), torch.zeros(1).to(self.device)

        start_of_replay = self.mb_x.shape[0]

        real_data_loss = self._criterion(
            self.noise_x[:start_of_replay], self.mb_output[:start_of_replay])
        replay_data_loss = self._criterion(
            self.noise_x[start_of_replay:], self.mb_output[start_of_replay:]) * self.lambd

        return real_data_loss, replay_data_loss

    def forward(self):
        raise NotImplementedError

    def training_epoch(self, **kwargs):
        """
        Training loop over the current `self.dataloader`.
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            batch_size = self.mbatch[0].shape[0]

            # Sample noise
            self.noise_x = torch.randn(self.mbatch[0].shape).to(self.device)
            self.timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device
            ).long()

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.data_loss, self.replay_loss = self.criterion()
            replay_weight = 1 - (1/(self.experience.current_experience+1))

            if self.weight_replay_loss:
                self.loss += (1 - replay_weight) * self.data_loss + \
                    replay_weight * self.replay_loss
            else:
                self.loss += self.data_loss + self.replay_loss

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _before_training_exp(self, **kwargs):
        """
        Called before the training of a new experience starts.
        """
        self.train_exp_id = self.experience.current_experience

        if self.untrained_generator:
            return super()._before_training_exp(**kwargs)

        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        return super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        self.untrained_generator = False
        return super()._after_training_exp(**kwargs)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            # self.mb_output = self.model.generate(self.mbatch[0].shape[0], output_type="torch")
            self.mb_output = None
            self._after_eval_forward(**kwargs)
            # self.loss = self.criterion()

            self._after_eval_iteration(**kwargs)

    @torch.no_grad()
    def eval(
        self,
        exp_list: Union[CLExperience, CLStream],
        **kwargs,
    ):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        # eval can be called inside the train method.
        # Save the shared state here to restore before returning.
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        if not isinstance(exp_list, Iterable):
            exp_list = [exp_list]
        self.current_eval_stream = exp_list

        self._before_eval(**kwargs)
        for self.experience in exp_list:
            # We don't need to evaluate experiences that have not been trained yet.
            if self.experience.current_experience > self.train_exp_id:
                continue
            self._before_eval_exp(**kwargs)
            self._eval_exp(**kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)

        # restore previous shared state.
        self._load_train_state(prev_train_state)


class NaiveDiffusionTraining(BaseDiffusionTraining):
    def __init__(self,
                 *args, **kwargs
                 ):
        super().__init__(weight_replay_loss=False, *args, **kwargs)

    def criterion(self):
        """
        Compute the loss for the current minibatch.
        """
        real_data_loss = self._criterion(self.noise_x, self.mb_output)
        return real_data_loss, torch.zeros(1).to(self.device)

    def forward(self):
        noisy_images = self.scheduler.add_noise(
            self.mbatch[0], self.noise_x, self.timesteps)
        return self.model(noisy_images, self.timesteps, return_dict=False)[0]


class CumulativeDiffusionTraining(NaiveDiffusionTraining):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None  # cumulative dataset

    def train_dataset_adaptation(self, **kwargs):
        """
        Concatenates all the previous experiences.
        """
        if self.dataset is None:
            self.dataset = self.experience.dataset
        else:
            self.dataset = concat_datasets(
                [self.dataset, self.experience.dataset]
            )
        self.adapted_dataset = self.dataset

        
class EWCDiffusionTraining(NaiveDiffusionTraining):
    def __init__(self,
                 ewc_lambda: float,
                 mode: str = "separate",
                 decay_factor: Optional[float] = None, 
                 keep_importance_data: bool = False,
                 *args, **kwargs):
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if kwargs["plugins"] is None:
            kwargs["plugins"] = [ewc]
        else:
            kwargs["plugins"].append(ewc)
        super().__init__(*args, **kwargs)

        
class SIDiffusionTraining(NaiveDiffusionTraining):
    def __init__(self,
                 si_lambda: Union[float, Sequence[float]],
                 eps: float = 0.0000001,
                 *args, **kwargs):
        if kwargs["plugins"] is None:
            kwargs["plugins"] = []
        kwargs["plugins"].append(SynapticIntelligencePlugin(si_lambda=si_lambda, eps=eps))
        super().__init__(*args, **kwargs)


class GaussianDistillationDiffusionTraining(BaseDiffusionTraining):
    def __init__(self,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def forward(self):
        batch_size = self.mbatch[0].shape[0]
        noisy_images = self.scheduler.add_noise(
            self.mbatch[0], self.noise_x, self.timesteps)

        if not self.untrained_generator:
            assert self.old_model is not None
            noise_replay = torch.randn(self.mbatch[0].shape).to(self.device)
            # # Below timestep 50, the loss impact is minimal. Ref: TODO
            timesteps_replay = torch.randint(
                self.replay_start_timestep, self.scheduler.config.num_train_timesteps, (
                    batch_size,), device=self.device
            ).long()
            with torch.no_grad():
                noise_prediction = self.old_model(
                    noise_replay, timesteps_replay, return_dict=False)[0]
            noisy_images = torch.cat([noisy_images, noise_replay], dim=0)
            self.noise_x = torch.cat([self.noise_x, noise_prediction], dim=0)
            self.timesteps = torch.cat(
                [self.timesteps, timesteps_replay], dim=0)

        return self.model(noisy_images, self.timesteps, return_dict=False)[0]


class GaussianSymmetryDistillationDiffusionTraining(BaseDiffusionTraining):
    def __init__(self,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def forward(self):
        batch_size = self.mbatch[0].shape[0]
        noisy_images = self.scheduler.add_noise(
            self.mbatch[0], self.noise_x, self.timesteps)

        if not self.untrained_generator:
            assert self.old_model is not None
            sample_size = self.mbatch[0].shape[-1]
            timesteps_replay = torch.randint(
                self.replay_start_timestep, self.scheduler.config.num_train_timesteps, (
                    batch_size,), device=self.device
            ).long()
            noise_replay = torch.randn(self.mbatch[0].shape).to(self.device)
            # Symmetrize noise
            t_666_333_idx = torch.where(
                (timesteps_replay <= 666) & (timesteps_replay > 333))[0]
            t_666_333_idx_h = t_666_333_idx[:len(t_666_333_idx)//2]
            t_666_333_idx_v = t_666_333_idx[len(t_666_333_idx)//2:]
            t_333_1_idx = torch.where(timesteps_replay <= 333)[0]
            noise_replay[t_666_333_idx_h, :, :sample_size//2, :] = torch.flip(
                noise_replay[t_666_333_idx_h, :, sample_size//2:, :], dims=[2,])
            noise_replay[t_666_333_idx_v, :, :, :sample_size//2] = torch.flip(
                noise_replay[t_666_333_idx_v, :, :, sample_size//2:], dims=[3,])
            noise_replay[t_333_1_idx, :, :sample_size//2, :] = torch.flip(
                noise_replay[t_333_1_idx, :, sample_size//2:, :], dims=[2,])
            noise_replay[t_333_1_idx, :, :, :sample_size//2] = torch.flip(
                noise_replay[t_333_1_idx, :, :, sample_size//2:], dims=[3,])
            with torch.no_grad():
                noise_prediction = self.old_model(
                    noise_replay, timesteps_replay, return_dict=False)[0]
            noisy_images = torch.cat([noisy_images, noise_replay], dim=0)
            self.noise_x = torch.cat([self.noise_x, noise_prediction], dim=0)
            self.timesteps = torch.cat(
                [self.timesteps, timesteps_replay], dim=0)

        return self.model(noisy_images, self.timesteps, return_dict=False)[0]


class LwFDistillationDiffusionTraining(BaseDiffusionTraining):
    def __init__(self,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def criterion(self):
        """
        Compute the loss for the current minibatch.
        """
        if self.experience.current_experience == 0 or not self.model.training:
            return self._criterion(self.noise_x, self.mb_output), torch.zeros(1).to(self.device)

        start_of_replay = self.mb_x.shape[0]

        real_data_loss = self._criterion(
            self.noise_x[:start_of_replay], self.mb_output)
        replay_data_loss = self._criterion(
            self.noise_x[start_of_replay:], self.mb_output) * self.lambd

        return real_data_loss, replay_data_loss

    def forward(self):
        noisy_images = self.scheduler.add_noise(
            self.mbatch[0], self.noise_x, self.timesteps)

        if not self.untrained_generator:
            assert self.old_model is not None
            with torch.no_grad():
                noise_prediction = self.old_model(
                    noisy_images, self.timesteps, return_dict=False)[0]
            self.noise_x = torch.cat([self.noise_x, noise_prediction], dim=0)

        return self.model(noisy_images, self.timesteps, return_dict=False)[0]


class FullGenerationDistillationDiffusionTraining(BaseDiffusionTraining):
    def __init__(self,
                 teacher_steps: int,
                 teacher_eta: float,
                 *args, **kwargs
                 ):
        self.teacher_steps = teacher_steps
        self.teacher_eta = teacher_eta
        super().__init__(*args, **kwargs)

    def forward(self):
        batch_size = self.mbatch[0].shape[0]
        noisy_images = self.scheduler.add_noise(
            self.mbatch[0], self.noise_x, self.timesteps)

        if not self.untrained_generator:
            assert self.old_model is not None
            noise_replay = torch.randn(self.mbatch[0].shape).to(self.device)
            timesteps_replay = torch.randint(
                self.replay_start_timestep, self.scheduler.config.num_train_timesteps, (
                    batch_size,), device=self.device
            ).long()

            replay_images = self.old_model.generate(
                batch_size, generation_steps=self.teacher_steps, eta=self.teacher_eta)
            noisy_replay_images = self.scheduler.add_noise(
                replay_images, noise_replay, timesteps_replay)

            with torch.no_grad():
                noise_prediction = self.old_model(
                    noisy_replay_images, timesteps_replay, return_dict=False)[0]

            noisy_images = torch.cat(
                [noisy_images, noisy_replay_images], dim=0)
            self.noise_x = torch.cat([self.noise_x, noise_prediction], dim=0)
            self.timesteps = torch.cat(
                [self.timesteps, timesteps_replay], dim=0)

        return self.model(noisy_images, self.timesteps, return_dict=False)[0]


class PartialGenerationDistillationDiffusionTraining(BaseDiffusionTraining):
    def __init__(self,
                 teacher_steps: int,
                 teacher_eta: float,
                 *args, **kwargs
                 ):
        self.teacher_steps = teacher_steps
        self.teacher_eta = teacher_eta
        super().__init__(*args, **kwargs)

    def forward(self):
        batch_size = self.mbatch[0].shape[0]
        noisy_images = self.scheduler.add_noise(
            self.mbatch[0], self.noise_x, self.timesteps)

        if not self.untrained_generator:
            assert self.old_model is not None
            timesteps_replay = torch.randint(
                self.replay_start_timestep, self.scheduler.config.num_train_timesteps, (
                    batch_size,), device=self.device
            ).long()

            replay_images = self.old_model.generate(
                batch_size, timesteps_replay, generation_steps=self.teacher_steps, eta=self.teacher_eta)

            with torch.no_grad():
                noise_prediction = self.old_model(
                    replay_images, timesteps_replay, return_dict=False)[0]

            noisy_images = torch.cat([noisy_images, replay_images], dim=0)
            self.noise_x = torch.cat([self.noise_x, noise_prediction], dim=0)
            self.timesteps = torch.cat(
                [self.timesteps, timesteps_replay], dim=0)

        return self.model(noisy_images, self.timesteps, return_dict=False)[0]


class NoDistillationDiffusionTraining(BaseDiffusionTraining):
    def __init__(self,
                 teacher_steps: int,
                 teacher_eta: float,
                 *args, **kwargs
                 ):
        self.teacher_steps = teacher_steps
        self.teacher_eta = teacher_eta
        super().__init__(*args, **kwargs)

    def forward(self):
        batch_size = self.mbatch[0].shape[0]
        noisy_images = self.scheduler.add_noise(
            self.mbatch[0], self.noise_x, self.timesteps)

        if not self.untrained_generator:
            assert self.old_model is not None
            noise_replay = torch.randn(self.mbatch[0].shape).to(self.device)
            timesteps_replay = torch.randint(
                self.replay_start_timestep, self.scheduler.config.num_train_timesteps, (
                    batch_size,), device=self.device
            ).long()

            replay_images = self.old_model.generate(
                batch_size, generation_steps=self.teacher_steps, eta=self.teacher_eta)
            noisy_replay_images = self.scheduler.add_noise(
                replay_images, noise_replay, timesteps_replay)

            noisy_images = torch.cat(
                [noisy_images, noisy_replay_images], dim=0)
            self.noise_x = torch.cat([self.noise_x, noise_replay], dim=0)
            self.timesteps = torch.cat(
                [self.timesteps, timesteps_replay], dim=0)

        return self.model(noisy_images, self.timesteps, return_dict=False)[0]


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
        evaluator: EvaluationPlugin = get_default_generator_logger(),
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

        # TODO: This only works for replay_size = None...
        start_of_replay = self.mb_x.shape[0] // 2

        mb_output_real = (
            self.mb_output[0][:start_of_replay],
            self.mb_output[1][:start_of_replay],
            self.mb_output[2][:start_of_replay],
        )
        mb_output_replay = (
            self.mb_output[0][start_of_replay:],
            self.mb_output[1][start_of_replay:],
            self.mb_output[2][start_of_replay:],
        )
        real_data_loss = self._criterion(
            self.mb_x[:start_of_replay], mb_output_real)
        replay_data_loss = self._criterion(
            self.mb_x[start_of_replay:], mb_output_replay)
        return ((1/(self.experience.current_experience+1)) * real_data_loss
                + (1 - (1/(self.experience.current_experience+1))) * replay_data_loss)
