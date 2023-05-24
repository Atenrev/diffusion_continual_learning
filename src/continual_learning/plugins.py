from copy import deepcopy
from avalanche.core import SupervisedPlugin
import torch


class EfficientGenerativeReplayPlugin(SupervisedPlugin):
    """
    Experience generative replay plugin.

    Updates the current mbatch of a strategy before training an experience
    by sampling a generator model and concatenating the replay data to the
    current batch.

    In this version of the plugin the number of replay samples is
    increased with each new experience. Another way to implempent
    the algorithm is by weighting the loss function and give more
    importance to the replayed data as the number of experiences
    increases. This will be implemented as an option for the user soon.

    :param generator_strategy: In case the plugin is applied to a non-generative
     model (e.g. a simple classifier), this should contain an Avalanche strategy
     for a model that implements a 'generate' method
     (see avalanche.models.generator.Generator). Defaults to None.
    :param untrained_solver: if True we assume this is the beginning of
        a continual learning task and add replay data only from the second
        experience onwards, otherwise we sample and add generative replay data
        before training the first experience. Default to True.
    :param replay_size: The user can specify the batch size of replays that
        should be added to each data batch. By default each data batch will be
        matched with replays of the same number.
    :param increasing_replay_size: If set to True, each experience this will
        double the amount of replay data added to each data batch. The effect
        will be that the older experiences will gradually increase in importance
        to the final loss.
    """

    def __init__(
        self,
        generator_strategy=None,
        untrained_solver: bool = True,
        replay_size: int = None,
        increasing_replay_size: bool = False,
    ):
        """
        Init.
        """
        super().__init__()
        self.generator_strategy = generator_strategy
        if self.generator_strategy:
            self.generator = generator_strategy.model
        else:
            self.generator = None
        self.untrained_solver = untrained_solver
        self.model_is_generator = False
        self.replay_size = replay_size
        self.increasing_replay_size = increasing_replay_size

    def before_training(self, strategy, *args, **kwargs):
        """Checks whether we are using a user defined external generator
        or we use the strategy's model as the generator.
        If the generator is None after initialization
        we assume that strategy.model is the generator.
        (e.g. this would be the case when training a VAE with
        generative replay)"""
        if not self.generator_strategy:
            self.generator_strategy = strategy
            self.generator = strategy.model
            self.model_is_generator = True

    def before_training_exp(
        self, strategy, num_workers: int = 0, shuffle: bool = True, **kwargs
    ):
        """
        Make deep copies of generator and solver before training new experience.
        Then, generate replay data and store it in the strategy's replay buffer.
        """
        if self.untrained_solver:
            # The solver needs to be trained before labelling generated data and
            # the generator needs to be trained before we can sample.
            return
        self.old_generator = deepcopy(self.generator)
        self.old_generator.eval()
        if not self.model_is_generator:
            self.old_model = deepcopy(strategy.model)
            self.old_model.eval()

        if self.untrained_solver:
            # The solver needs to be trained before labelling generated data and
            # the generator needs to be trained before we can sample.
            return

        if self.replay_size:
            number_replays_to_generate = self.replay_size
        else:
            if self.increasing_replay_size:
                number_replays_to_generate = self.generator_strategy.train_mb_size * (
                    strategy.experience.current_experience
                ) + self.generator_strategy.train_mb_size
            else:
                number_replays_to_generate = self.generator_strategy.train_mb_size * 2

        print("Generating replay data...")
        with torch.no_grad():
            # Generate replay data in batches of max size train_mb_size
            # and concatenate them to a single replay buffer
            for i in range(
                number_replays_to_generate // self.generator_strategy.train_mb_size
            ):
                if i == 0:
                    self.replay_buffer = self.old_generator.generate(
                        self.generator_strategy.train_mb_size
                    ).to(strategy.device)
                else:
                    self.replay_buffer = torch.cat(
                        [
                            self.replay_buffer,
                            self.old_generator.generate(
                                self.generator_strategy.train_mb_size
                            ).to(strategy.device),
                        ],
                        dim=0,
                    )

            # self.replay_buffer = self.old_generator.generate(number_replays_to_generate).to(
            #     strategy.device
            # )

        print("Labelling replay data...")
        # extend y with predicted labels (or mock labels if model==generator)
        if not self.model_is_generator:
            with torch.no_grad():
                self.replay_output = self.old_model(self.replay_buffer).argmax(dim=-1)
        else:
            # Mock labels:
            self.replay_output = torch.zeros(self.replay_buffer.shape[0])

        self.replay_output = self.replay_output.to(strategy.device)

    def after_training_exp(
        self, strategy, num_workers: int = 0, shuffle: bool = True, **kwargs
    ):
        """
        Set untrained_solver boolean to False after (the first) experience,
        in order to start training with replay data from the second experience.
        """
        self.untrained_solver = False

    def before_training_iteration(self, strategy, **kwargs):
        """
        Appending replay data to current minibatch before
        each training iteration.
        """
        if self.untrained_solver:
            # The solver needs to be trained before labelling generated data and
            # the generator needs to be trained before we can sample.
            return
        
        batch_size = len(strategy.mbatch[0])
        replay_indices = torch.randperm(self.replay_buffer.shape[0])[:batch_size]
        
        strategy.mbatch[0] = torch.cat([strategy.mbatch[0], self.replay_buffer[replay_indices]], dim=0)
        
        strategy.mbatch[1] = torch.cat(
            [strategy.mbatch[1], self.replay_output[replay_indices]], dim=0
        )
        # extend task id batch (we implicitley assume a task-free case)
        strategy.mbatch[-1] = torch.cat(
            [
                strategy.mbatch[-1],
                torch.ones(self.replay_buffer.shape[0]).to(strategy.device)
                * strategy.mbatch[-1][0],
            ],
            dim=0,
        )


class TrainDiffusionGeneratorAfterExpPlugin(SupervisedPlugin):
    """
    TrainGeneratorAfterExpPlugin makes sure that after each experience of
    training the solver of a scholar model, we also train the generator on the
    data of the current experience.
    """

    def after_training_exp(self, strategy, **kwargs):
        """
        The training method expects an Experience object
        with a 'dataset' parameter.
        """
        for plugin in strategy.plugins:
            if type(plugin) is EfficientGenerativeReplayPlugin:
                plugin.generator_strategy.train(strategy.experience)
