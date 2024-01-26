from copy import deepcopy
from avalanche.core import SupervisedPlugin
import torch


class OldGeneratorManager:
    """
    OldGeneratorManager is a class that manages the old generator
    when using the UpdatedGenerativeReplayPlugin with multiple
    strategies. Stores the old generator as a singleton.
    """
    _old_generator = None
    _current_experience = None

    @classmethod
    def update_and_get_old_generator(cls, generator, current_experience):
        """
        Sets the old generator and the current experience.
        If the current experience is the same as the previous one,
        the old generator is not updated.

        Returns the old generator.
        """
        if cls._current_experience != current_experience:
            cls._current_experience = current_experience
            new_device = generator.device

            # Check if there is a second GPU available
            if torch.cuda.device_count() > 1:
                # If yes, move the generator to the second GPU
                new_device = torch.device(torch.device("cuda:1"))

            cls._old_generator = deepcopy(generator)
            cls._old_generator.to(new_device)
            cls._old_generator.eval()

        return cls._old_generator


class UpdatedGenerativeReplayPlugin(SupervisedPlugin):
    """
    Experience generative replay plugin.

    Updates the current mbatch of a strategy before training an experience
    by sampling a generator model and concatenating the replay data to the
    current batch.

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
    :param T: Temperature parameter for scaling logits of the replay data.
    """

    def __init__(
        self,
        generator_strategy=None,
        untrained_solver: bool = True,
        replay_size: int = None,
        increasing_replay_size: bool = False,
        # T: float = 2.0,
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
        """
        Checks whether we are using a user defined external generator
        or we use the strategy's model as the generator.
        If the generator is None after initialization
        we assume that strategy.model is the generator.
        (e.g. this would be the case when training a VAE with
        generative replay)
        """
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
            return
        
        self.old_generator = OldGeneratorManager.update_and_get_old_generator(
            self.generator, strategy.experience.current_experience
        )

        if not self.model_is_generator:
            self.old_model = deepcopy(strategy.model)

            if torch.cuda.device_count() > 1:
                self.old_model.to(torch.device(torch.device("cuda:1")))

            self.old_model.eval()

        torch.cuda.empty_cache()

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
            return
        
        self.current_batch_size = len(strategy.mbatch[0])
        
        if self.replay_size:
            number_replays_to_generate = self.replay_size
        else:
            if self.increasing_replay_size:
                number_replays_to_generate = len(strategy.mbatch[0]) * (
                    strategy.experience.current_experience
                )
            else:
                number_replays_to_generate = len(strategy.mbatch[0])
            
        replay = self.old_generator.generate(number_replays_to_generate).to(strategy.device)
        strategy.mbatch[0] = torch.cat([strategy.mbatch[0], replay], dim=0)

        # extend y with predicted labels (or mock labels if model==generator)
        if not self.model_is_generator:
            if torch.cuda.device_count() > 1:
                replay_in_old_device = replay.to(torch.device(torch.device("cuda:1")))
            else:
                replay_in_old_device = replay

            with torch.no_grad():
                replay_output = self.old_model(replay_in_old_device)
        else:
            # Mock labels:
            replay_output = torch.zeros(replay.shape[0])

        replay_output = replay_output.to(strategy.device)

        if replay_output.ndim > 1:
            # If we are using a classification model, we one-hot encode the labels
            # of the training data (so we can use soft labels for the replay data)
            strategy.mbatch[1] = torch.nn.functional.one_hot(
                        strategy.mbatch[1], num_classes=replay_output.shape[1]
                    ).to(strategy.device)
        
        # Then we append the replay data to the current minibatch
        strategy.mbatch[1] = torch.cat(
            [strategy.mbatch[1], replay_output], dim=0
        )

        # extend task id batch (we implicitly assume a task-free case)
        strategy.mbatch[-1] = torch.cat(
            [
                strategy.mbatch[-1],
                torch.ones(replay.shape[0]).to(strategy.device)
                * strategy.mbatch[-1][0],
            ],
            dim=0,
        )


class TrainGeneratorAfterExpPlugin(SupervisedPlugin):
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
            if type(plugin) is UpdatedGenerativeReplayPlugin:
                plugin.generator_strategy.train(strategy.experience)
