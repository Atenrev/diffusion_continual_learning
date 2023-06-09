import os
import argparse
import torch
import torch.optim.lr_scheduler
import datetime

from torchvision import transforms
from diffusers import UNet2DModel, DDIMScheduler
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    confusion_matrix_metrics,
)
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin

from src.continual_learning.strategies import WeightedSoftGenerativeReplay, DiffusionTraining
from src.continual_learning.plugins import UpdatedGenerativeReplayPlugin
from src.continual_learning.metrics import ExperienceFIDMetric
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.diffusion_utils import wrap_in_pipeline, evaluate_diffusion


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--train_timesteps", type=int, default=1000)
    parser.add_argument("--generation_steps", type=int, default=20)
    parser.add_argument("--eta", type=int, default=1.0)
    parser.add_argument("--epochs_generator", type=int, default=10)
    parser.add_argument("--epochs_solver", type=int, default=4)
    parser.add_argument("--generator_lr", type=float, default=1e-3)
    parser.add_argument("--solver_lr", type=float, default=0.001)
    parser.add_argument("--increasing_replay_size", type=bool, default=False)
    parser.add_argument("--replay_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument("--debug", action="store_true", default=True)
    return parser.parse_args()


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- BENCHMARK CREATION
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )
    benchmark = SplitMNIST(
        n_experiences=5, 
        seed=args.seed,
        train_transform=train_transform,
        eval_transform=train_transform,
    )

    run_name = "generative_replay_diffusion_classifier"
    run_name += f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if args.debug:
        project_name = "master-thesis-debug"
    else:
        project_name = "master-thesis"
    wandb_logger = WandBLogger(
        project_name=project_name, 
        run_name=run_name, 
        config=vars(args)
    )

    # --- GENERATOR MODEL CREATION
    generator_model = UNet2DModel(
        sample_size=args.image_size,  # the target image resolution
        in_channels=args.channels,  # the number of input channels, 3 for RGB images
        out_channels=args.channels,  # the number of output channels
        layers_per_block=1,  # how many ResNet layers to use per UNet block
        block_out_channels=(16, 32, 32, 64),
        norm_num_groups=16,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    noise_scheduler = DDIMScheduler(num_train_timesteps=args.train_timesteps)
    wrap_in_pipeline(generator_model, noise_scheduler, DDIMPipeline, args.generation_steps, args.eta) 

    gen_eval_plugin = EvaluationPlugin(
        ExperienceFIDMetric(),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loggers=[wandb_logger, InteractiveLogger()],
    )

    # CREATE THE GENERATOR STRATEGY INSTANCE (DiffusionTraining)
    generator_strategy = DiffusionTraining(
        generator_model,
        noise_scheduler,
        torch.optim.Adam(generator_model.parameters(), lr=args.generator_lr),
        train_mb_size=args.batch_size,
        train_epochs=args.epochs_generator,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=gen_eval_plugin,
        train_timesteps=args.train_timesteps,
        plugins=[
            UpdatedGenerativeReplayPlugin(
                increasing_replay_size=args.increasing_replay_size,
                replay_size=args.replay_size,
            )
        ],
    )   

    # --- MODEL CREATION
    model = SimpleMLP(
        input_size=args.image_size * args.image_size,
        num_classes=benchmark.n_classes
    )

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(
            stream=True, wandb=True, class_names=[str(i) for i in range(10)]
        ),
        loggers=[wandb_logger],
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = WeightedSoftGenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=args.solver_lr),
        CrossEntropyLoss(),
        train_mb_size=args.batch_size, # Caution: the batch size is doubled because of the replay
        train_epochs=args.epochs_solver,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=eval_plugin,
        generator_strategy=generator_strategy,
        increasing_replay_size=args.increasing_replay_size,
        replay_size=args.replay_size,
    )

    # OUTPUT DIRECTORY
    output_dir = f"results/generative_replay/diffusion_classifier/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    n_samples = 20
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))
        # generator_strategy.eval(benchmark.test_stream)
        
        print("Computing generated samples and saving them to disk")
        pipeline = DDIMPipeline(unet=generator_model, scheduler=noise_scheduler)
        evaluate_diffusion(output_dir, n_samples, experience.current_experience, pipeline, steps=args.generation_steps, seed=args.seed, eta=args.eta)

    print("Evaluation completed")        


if __name__ == "__main__":
    args = __parse_args()
    main(args)