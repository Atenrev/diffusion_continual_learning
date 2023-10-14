import unittest
import torch

from diffusers import UNet2DModel
from torch.nn import CrossEntropyLoss, MSELoss

from src.models.simple_cnn import SimpleCNN
from src.schedulers.scheduler_ddim import DDIMScheduler
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.common.diffusion_utils import wrap_in_pipeline
from src.continual_learning.strategies import (
    WeightedSoftGenerativeReplay,
    NoDistillationDiffusionTraining,
    LwFDistillationDiffusionTraining,
)


SOLVER_CONFIG = {
    "model": {
        "channels": 1,
        "n_classes": 10
    },
    "optimizer": {
        "lr": 0.001
    },
    "strategy": {
        "train_batch_size": 32,
        "epochs": 1,
        "eval_batch_size": 32,
        "replay_size": 20,
        "increasing_replay_size": False
    }
}
GENERATOR_CONFIG = {
    "model": {
        "name": "unet2d",
        "input_size": 32,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": [32, 64],
        "norm_num_groups": 32,
        "down_block_types": ["DownBlock2D","DownBlock2D"],
        "up_block_types": ["UpBlock2D","UpBlock2D"]
    },
    "scheduler": {
        "name": "DDIM",
        "train_timesteps": 1000
    },
    "optimizer": {
        "name": "adam",
        "lr": 0.0002
    },
    "strategy": {
        "teacher_steps": 10,
        "teacher_eta": 0.0,
        "train_batch_size": 32,
        "epochs": 1,
        "eval_batch_size": 32,
        "replay_size": 20,
        "increasing_replay_size": False,
        "replay_start_timestep": 0,
        "weight_replay_loss": True,
    }
}
DEVICE = "cpu"


def get_generator_model():
    generator_model = UNet2DModel(
            sample_size=GENERATOR_CONFIG["model"]["input_size"],
            in_channels=GENERATOR_CONFIG["model"]["in_channels"],
            out_channels=GENERATOR_CONFIG["model"]["out_channels"],
            layers_per_block=GENERATOR_CONFIG["model"]["layers_per_block"],
            block_out_channels=GENERATOR_CONFIG["model"]["block_out_channels"],
            norm_num_groups=GENERATOR_CONFIG["model"]["norm_num_groups"],
            down_block_types=GENERATOR_CONFIG["model"]["down_block_types"],
            up_block_types=GENERATOR_CONFIG["model"]["up_block_types"],
        )
    noise_scheduler = DDIMScheduler(num_train_timesteps=GENERATOR_CONFIG["scheduler"]["train_timesteps"])
    wrap_in_pipeline(generator_model, noise_scheduler,
                        DDIMPipeline, 2, 0.0, def_output_type="torch_raw")
    return generator_model, noise_scheduler


class WeightedSoftGenerativeReplayTests(unittest.TestCase):
    def setUp(self):
        solver_model = SimpleCNN(
            n_channels=SOLVER_CONFIG["model"]["channels"],
            num_classes=SOLVER_CONFIG["model"]["n_classes"],
        )
        generator_model, noise_scheduler = get_generator_model()

        self.generator_strategy = NoDistillationDiffusionTraining(
            GENERATOR_CONFIG["strategy"]["teacher_steps"],
            GENERATOR_CONFIG["strategy"]["teacher_eta"],
            model=generator_model,
            scheduler=noise_scheduler,
            optimizer=torch.optim.Adam(generator_model.parameters(),
                            lr=GENERATOR_CONFIG["optimizer"]["lr"]),
            train_mb_size=GENERATOR_CONFIG["strategy"]["train_batch_size"],
            train_epochs=GENERATOR_CONFIG["strategy"]["epochs"],
            eval_mb_size=GENERATOR_CONFIG["strategy"]["eval_batch_size"],
            device=DEVICE,
            train_timesteps=GENERATOR_CONFIG["scheduler"]["train_timesteps"],
            lambd=1.0,
            replay_start_timestep=GENERATOR_CONFIG["strategy"]["replay_start_timestep"],
            weight_replay_loss=GENERATOR_CONFIG["strategy"]["weight_replay_loss"],
        )
        self.solver_strategy = WeightedSoftGenerativeReplay(
            solver_model,
            torch.optim.Adam(solver_model.parameters(), lr=SOLVER_CONFIG["optimizer"]["lr"]),
            CrossEntropyLoss(),
            train_mb_size=SOLVER_CONFIG["strategy"]["train_batch_size"],
            train_epochs=SOLVER_CONFIG["strategy"]["epochs"],
            eval_mb_size=SOLVER_CONFIG["strategy"]["eval_batch_size"],
            device=DEVICE,
            plugins=None,
            generator_strategy=self.generator_strategy,
            increasing_replay_size=SOLVER_CONFIG["strategy"]["increasing_replay_size"],
            replay_size=SOLVER_CONFIG["strategy"]["replay_size"],
        )

    def test_criterion(self):
        mb_x = torch.randn(32, 1, 32, 32)
        mb_y = torch.randint(0, 10, (32,))
        mb_output = torch.randn(32, 10)
        cross_entropy_loss_true = CrossEntropyLoss()(mb_output, mb_y)

        self.solver_strategy.experience = unittest.mock.Mock()
        self.solver_strategy.experience.current_experience = 0
        self.solver_strategy.mb_output = mb_output
        self.solver_strategy.mbatch = (mb_x, mb_y)

        assert self.solver_strategy.criterion() == cross_entropy_loss_true

    def test_criterion_with_replay(self):
        mb_x = torch.randn(64, 1, 32, 32)
        mb_y = torch.randint(0, 10, (32,))
        replay_output = torch.randn(32, 10)
        mb_y_onehot = torch.zeros((32, 10))
        mb_y_onehot[range(32), mb_y] = 1
        mb_y = torch.cat([mb_y_onehot, replay_output], dim=0)
        mb_output = torch.randn(64, 10)

        loss_current = CrossEntropyLoss()(mb_output[:32], mb_y[:32])
        mb_y_replay = mb_y[32:]
        # Scale logits by temperature according to M. van de Ven et al. (2020)
        mb_y_replay = mb_y_replay / self.solver_strategy.T
        mb_y_replay = mb_y_replay.log_softmax(dim=1)
        output_replay = mb_output[32:]
        output_replay = output_replay / self.solver_strategy.T
        output_replay = output_replay.softmax(dim=1)
        loss_replay = -output_replay * mb_y_replay
        loss_replay = loss_replay.sum(dim=1).mean()
        loss_replay = loss_replay * self.solver_strategy.T**2
        
        self.solver_strategy.experience = unittest.mock.Mock()
        self.solver_strategy.mb_output = mb_output
        self.solver_strategy.mbatch = (mb_x, mb_y)

        self.solver_strategy.experience.current_experience = 1
        assert (self.solver_strategy.criterion() == (1/2) * loss_current 
            + (1/2) * loss_replay)

        self.solver_strategy.experience.current_experience = 2
        assert (self.solver_strategy.criterion() == (1/3) * loss_current 
            + (2/3) * loss_replay)
        

class NoDistillationDiffusionTrainingTests(unittest.TestCase):
    def setUp(self):
        generator_model, noise_scheduler = get_generator_model()
        self.generator_strategy = NoDistillationDiffusionTraining(
            GENERATOR_CONFIG["strategy"]["teacher_steps"],
            GENERATOR_CONFIG["strategy"]["teacher_eta"],
            criterion=MSELoss(),
            model=generator_model,
            scheduler=noise_scheduler,
            optimizer=torch.optim.Adam(generator_model.parameters(),
                            lr=GENERATOR_CONFIG["optimizer"]["lr"]),
            train_mb_size=GENERATOR_CONFIG["strategy"]["train_batch_size"],
            train_epochs=GENERATOR_CONFIG["strategy"]["epochs"],
            eval_mb_size=GENERATOR_CONFIG["strategy"]["eval_batch_size"],
            device=DEVICE,
            train_timesteps=GENERATOR_CONFIG["scheduler"]["train_timesteps"],
            lambd=1.0,
            replay_start_timestep=GENERATOR_CONFIG["strategy"]["replay_start_timestep"],
            weight_replay_loss=GENERATOR_CONFIG["strategy"]["weight_replay_loss"],
        )

    def test_criterion(self):
        mb_x = torch.randn(32, 1, 32, 32)
        mb_y = torch.randint(0, 10, (32,))
        noise_x = torch.randn(32, 1, 32, 32)
        mb_output = torch.randn(32, 1, 32, 32)
        mse_loss_true = MSELoss()(mb_output, noise_x)

        self.generator_strategy.experience = unittest.mock.Mock()
        self.generator_strategy.model.training = True
        self.generator_strategy.mbatch = (mb_x, mb_y)
        self.generator_strategy.mb_output = mb_output
        self.generator_strategy.noise_x = noise_x

        self.generator_strategy.lambd = 1.0
        self.generator_strategy.experience.current_experience = 0
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == mse_loss_true
        assert loss_replay_predicted == 0

        self.generator_strategy.lambd = 0.0
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == mse_loss_true
        assert loss_replay_predicted == 0

    def test_criterion_with_replay(self):
        mb_x = torch.randn(32, 1, 32, 32)
        mb_y = torch.randint(0, 10, (32,))
        noise_x = torch.randn(32, 1, 32, 32)
        noise_replay = torch.randn(32, 1, 32, 32)
        noise_x = torch.cat([noise_x, noise_replay], dim=0)
        mb_output = torch.randn(64, 1, 32, 32)

        loss_current = MSELoss()(mb_output[:32], noise_x[:32])
        loss_replay = MSELoss()(mb_output[32:], noise_x[32:])

        self.generator_strategy.experience = unittest.mock.Mock()
        self.generator_strategy.model.training = True
        self.generator_strategy.mbatch = (mb_x, mb_y)
        self.generator_strategy.mb_output = mb_output
        self.generator_strategy.noise_x = noise_x

        self.generator_strategy.lambd = 2.0
        self.generator_strategy.experience.current_experience = 1
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == loss_current
        assert loss_replay_predicted == loss_replay * 2.0
        
        self.generator_strategy.lambd = 0.0
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == loss_current
        assert loss_replay_predicted == loss_replay * 0.0

        self.generator_strategy.lambd = 2.0
        self.generator_strategy.experience.current_experience = 2
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == loss_current
        assert loss_replay_predicted == loss_replay * 2.0        

        self.generator_strategy.lambd = 0.0
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == loss_current
        assert loss_replay_predicted == loss_replay * 0.0


class LwFDistillationDiffusionTrainingTests(unittest.TestCase):
    def setUp(self):
        generator_model, noise_scheduler = get_generator_model()
        self.generator_strategy = LwFDistillationDiffusionTraining(
            criterion=MSELoss(),
            model=generator_model,
            scheduler=noise_scheduler,
            optimizer=torch.optim.Adam(generator_model.parameters(),
                            lr=GENERATOR_CONFIG["optimizer"]["lr"]),
            train_mb_size=GENERATOR_CONFIG["strategy"]["train_batch_size"],
            train_epochs=GENERATOR_CONFIG["strategy"]["epochs"],
            eval_mb_size=GENERATOR_CONFIG["strategy"]["eval_batch_size"],
            device=DEVICE,
            train_timesteps=GENERATOR_CONFIG["scheduler"]["train_timesteps"],
            lambd=1.0,
            replay_start_timestep=GENERATOR_CONFIG["strategy"]["replay_start_timestep"],
            weight_replay_loss=GENERATOR_CONFIG["strategy"]["weight_replay_loss"],
        )

    def test_criterion(self):
        mb_x = torch.randn(32, 1, 32, 32)
        mb_y = torch.randint(0, 10, (32,))
        noise_x = torch.randn(32, 1, 32, 32)
        mb_output = torch.randn(32, 1, 32, 32)
        mse_loss_true = MSELoss()(mb_output, noise_x)

        self.generator_strategy.experience = unittest.mock.Mock()
        self.generator_strategy.model.training = True
        self.generator_strategy.mbatch = (mb_x, mb_y)
        self.generator_strategy.mb_output = mb_output
        self.generator_strategy.noise_x = noise_x

        self.generator_strategy.lambd = 1.0
        self.generator_strategy.experience.current_experience = 0
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == mse_loss_true
        assert loss_replay_predicted == 0

        self.generator_strategy.lambd = 0.0
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == mse_loss_true
        assert loss_replay_predicted == 0

    def test_criterion_with_replay(self):
        mb_x = torch.randn(32, 1, 32, 32)
        mb_y = torch.randint(0, 10, (32,))
        noise_x = torch.randn(32, 1, 32, 32)
        noise_replay = torch.randn(32, 1, 32, 32)
        noise_x = torch.cat([noise_x, noise_replay], dim=0)
        mb_output = torch.randn(32, 1, 32, 32)

        loss_current = MSELoss()(mb_output, noise_x[:32])
        loss_replay = MSELoss()(mb_output, noise_x[32:])

        self.generator_strategy.experience = unittest.mock.Mock()
        self.generator_strategy.model.training = True
        self.generator_strategy.mbatch = (mb_x, mb_y)
        self.generator_strategy.mb_output = mb_output
        self.generator_strategy.noise_x = noise_x

        self.generator_strategy.lambd = 2.0
        self.generator_strategy.experience.current_experience = 1
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == loss_current
        assert loss_replay_predicted == loss_replay * 2.0

        self.generator_strategy.lambd = 0.0
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == loss_current
        assert loss_replay_predicted == loss_replay * 0.0

        self.generator_strategy.lambd = 2.0
        self.generator_strategy.experience.current_experience = 2
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == loss_current
        assert loss_replay_predicted == loss_replay * 2.0

        self.generator_strategy.lambd = 0.0
        loss_current_predicted, loss_replay_predicted = self.generator_strategy.criterion()
        assert loss_current_predicted == loss_current
        assert loss_replay_predicted == loss_replay * 0.0
        

if __name__ == "__main__":
    unittest.main()