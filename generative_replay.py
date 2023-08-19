import os
import argparse
import torch
import torch.optim.lr_scheduler
import datetime

from torchvision import transforms
from diffusers import UNet2DModel, DDIMScheduler
from torch.nn import CrossEntropyLoss

from avalanche.benchmarks import SplitMNIST, SplitFMNIST
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    confusion_matrix_metrics,
)
from avalanche.logging import WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage

from src.continual_learning.strategies import (
    WeightedSoftGenerativeReplay, 
    GaussianDistillationDiffusionTraining, 
    GaussianSymmetryDistillationDiffusionTraining,
    LwFDistillationDiffusionTraining,
    FullGenerationDistillationDiffusionTraining,
    PartialGenerationDistillationDiffusionTraining,
    NoDistillationDiffusionTraining,
    VAETraining
)
from src.continual_learning.plugins import UpdatedGenerativeReplayPlugin
from src.continual_learning.metrics.diffusion_metrics import DiffusionMetricsMetric
from src.continual_learning.metrics.loss import loss_metrics, replay_loss_metrics, data_loss_metrics
from src.continual_learning.loggers import TextLogger, CSVLogger
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.common.utils import get_configuration
from src.common.diffusion_utils import wrap_in_pipeline, evaluate_diffusion
from src.models.vae import MlpVAE, VAE_loss
from src.models.simple_cnn import SimpleCNN


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="split_fmnist")

    parser.add_argument("--generator_type", type=str, default="diffusion")
    parser.add_argument("--generator_config_path", type=str,
                        default="configs/model/ddim_medium.json")
    parser.add_argument("--generator_strategy_config_path",
                        type=str, default="configs/strategy/diffusion_debug.json")
    
    parser.add_argument("--lambd", type=float, default=1.0)
    parser.add_argument("--generation_steps", type=int, default=10) # Used in the solver strategy
    parser.add_argument("--eta", type=float, default=0.0)
    
    parser.add_argument("--solver_type", type=str, default="cnn")
    parser.add_argument("--solver_config_path", type=str,
                        default="configs/model/cnn.json")
    parser.add_argument("--solver_strategy_config_path", type=str,
                        default="configs/strategy/cnn_w_diffusion_debug.json")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument("--output_dir", type=str,
                        default="results_fuji/smasipca/generative_replay_debug/")
    parser.add_argument("--project_name", type=str, default="master-thesis-genreplay")
    parser.add_argument("--wandb", action="store_true", default=False)
    return parser.parse_args()


def get_benchmark(image_size: int, seed: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset == "split_fmnist":
        benchmark = SplitFMNIST(
            n_experiences=5,
            seed=seed,
            train_transform=train_transform,
            eval_transform=train_transform,
        )
    elif args.dataset == "split_mnist":
        benchmark = SplitMNIST(
            n_experiences=5,
            seed=seed,
            train_transform=train_transform,
            eval_transform=train_transform,
        )
    else:
        raise NotImplementedError(
            f"Dataset {args.dataset} not implemented")
    
    return benchmark


def get_generator_strategy(generator_type: str, model_config, strategy_config, loggers, device, generation_steps: int = 20, eta: float = 0.0, lambd: float = 1.0, checkpoint_plugin=None):
    generator_strategy = None

    plugins = []
    if checkpoint_plugin is not None:
        plugins.append(checkpoint_plugin)

    if generator_type == "diffusion":
        generator_model = UNet2DModel(
            sample_size=model_config.model.input_size,
            in_channels=model_config.model.in_channels,
            out_channels=model_config.model.out_channels,
            layers_per_block=model_config.model.layers_per_block,
            block_out_channels=model_config.model.block_out_channels,
            norm_num_groups=model_config.model.norm_num_groups,
            down_block_types=model_config.model.down_block_types,
            up_block_types=model_config.model.up_block_types,
        )
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=model_config.scheduler.train_timesteps)
        wrap_in_pipeline(generator_model, noise_scheduler,
                         DDIMPipeline, generation_steps, eta, def_output_type="torch_raw")
        gen_eval_plugin = EvaluationPlugin(
            loss_metrics(
                minibatch=True,
                epoch=True,
                epoch_running=True,
                experience=True,
                stream=True,
            ),
            replay_loss_metrics(
                minibatch=True,
                epoch=True,
                epoch_running=True,
                experience=True,
                stream=True,
            ),
            data_loss_metrics(
                minibatch=True,
                epoch=True,
                epoch_running=True,
                experience=True,
                stream=True,
            ),
            DiffusionMetricsMetric(),
            loggers=loggers,
        )

        if strategy_config.strategy == "full_generation_distillation":
            generator_strategy = FullGenerationDistillationDiffusionTraining(
                strategy_config.teacher_steps,
                strategy_config.teacher_eta,
                model=generator_model,
                scheduler=noise_scheduler,
                optimizer=torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                plugins=plugins,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                lambd=lambd,
                replay_start_timestep=strategy_config.replay_start_timestep,
                weight_replay_loss=strategy_config.weight_replay_loss,
            )
        elif strategy_config.strategy == "partial_generation_distillation":
            generator_strategy = PartialGenerationDistillationDiffusionTraining(
                strategy_config.teacher_steps,
                strategy_config.teacher_eta,
                model=generator_model,
                scheduler=noise_scheduler,
                optimizer=torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                plugins=plugins,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                lambd=lambd,
                replay_start_timestep=strategy_config.replay_start_timestep,
                weight_replay_loss=strategy_config.weight_replay_loss,
            )
        elif strategy_config.strategy == "no_distillation":
            generator_strategy = NoDistillationDiffusionTraining(
                strategy_config.teacher_steps,
                strategy_config.teacher_eta,
                model=generator_model,
                scheduler=noise_scheduler,
                optimizer=torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                plugins=plugins,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                lambd=lambd,
                replay_start_timestep=strategy_config.replay_start_timestep,
                weight_replay_loss=strategy_config.weight_replay_loss,
            )
        elif strategy_config.strategy == "gaussian_distillation":
            generator_strategy = GaussianDistillationDiffusionTraining(
                generator_model,
                noise_scheduler,
                torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                plugins=plugins,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                lambd=lambd,
                replay_start_timestep=strategy_config.replay_start_timestep,
                weight_replay_loss=strategy_config.weight_replay_loss,
            )
        elif strategy_config.strategy == "gaussian_symmetry_distillation":
            generator_strategy = GaussianSymmetryDistillationDiffusionTraining(
                generator_model,
                noise_scheduler,
                torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                plugins=plugins,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                lambd=lambd,
                replay_start_timestep=strategy_config.replay_start_timestep,
                weight_replay_loss=strategy_config.weight_replay_loss,
            )
        elif strategy_config.strategy == "lwf_distillation":
            generator_strategy = LwFDistillationDiffusionTraining(
                generator_model,
                noise_scheduler,
                torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                plugins=plugins,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                lambd=lambd,
                replay_start_timestep=strategy_config.replay_start_timestep,
                weight_replay_loss=strategy_config.weight_replay_loss,
            )
        else:
            raise NotImplementedError(
                f"Strategy {strategy_config.strategy} not implemented")

    elif generator_type == "vae":
        generator = MlpVAE(
            (model_config.model.channels, model_config.model.input_size,
             model_config.model.input_size),
            encoder_dims=model_config.model.encoder_dims,
            decoder_dims=model_config.model.decoder_dims,
            latent_dim=model_config.model.latent_dim,
            n_classes=model_config.model.n_classes,
            device=device
        )
        optimizer_generator = torch.optim.Adam(
            generator.parameters(),
            lr=model_config.optimizer.lr,
            betas=(0.9, 0.999),
        )
        gen_eval_plugin = EvaluationPlugin(
            DiffusionMetricsMetric(),
            loggers=loggers,
        )
        plugins.append(UpdatedGenerativeReplayPlugin(
                    replay_size=strategy_config.increasing_replay_size,
                    increasing_replay_size=strategy_config.increasing_replay_size,
                ))
        generator_strategy = VAETraining(
            model=generator,
            optimizer=optimizer_generator,
            criterion=VAE_loss,
            train_mb_size=strategy_config.train_batch_size,
            train_epochs=strategy_config.epochs,
            eval_mb_size=strategy_config.eval_batch_size,
            device=device,
            evaluator=gen_eval_plugin,
            plugins=plugins,
        )
    else:
        raise NotImplementedError(
            f"Generator type {generator_type} not implemented")

    return generator_strategy


def get_solver_strategy(solver_type: str, model_config, strategy_config, generator_strategy, loggers, device, checkpoint_plugin=None):
    if solver_type == "mlp":
        model = SimpleMLP(
            input_size=model_config.model.input_size *
            model_config.model.input_size * model_config.model.channels,
            num_classes=model_config.model.n_classes
        )
    elif solver_type == "cnn":
        model = SimpleCNN(
            n_channels=model_config.model.channels,
            num_classes=model_config.model.n_classes
        )
    else:
        raise NotImplementedError(
            f"Solver type {solver_type} not implemented")

    plugins = []
    if checkpoint_plugin is not None:
        plugins.append(checkpoint_plugin)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        # loss_metrics(
        #     minibatch=True,
        #     epoch=True,
        #     epoch_running=True,
        #     experience=True,
        #     stream=True,
        # ),
        forgetting_metrics(experience=True, stream=True),
        # confusion_matrix_metrics(
        #     stream=True, wandb=True, class_names=[str(i) for i in range(10)]
        # ),
        loggers=loggers,
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = WeightedSoftGenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=model_config.optimizer.lr),
        CrossEntropyLoss(),
        # Caution: the batch size is doubled because of the replay
        train_mb_size=strategy_config.train_batch_size,
        train_epochs=strategy_config.epochs,
        eval_mb_size=strategy_config.eval_batch_size,
        device=device,
        plugins=plugins,
        evaluator=eval_plugin,
        generator_strategy=generator_strategy,
        increasing_replay_size=strategy_config.increasing_replay_size,
        replay_size=strategy_config.replay_size,
    )

    return cl_strategy


def run_experiment(args, seed: int, device: torch.device):
    # --- SEEDING
    RNGManager.set_random_seeds(seed)
    torch.backends.cudnn.deterministic = True

    generator_config = get_configuration(args.generator_config_path)
    generator_strategy_config = get_configuration(args.generator_strategy_config_path)
    run_name = f"gr_{args.generator_type}_{generator_strategy_config.strategy}_steps_{args.generation_steps}_lambd_{args.lambd}_{args.solver_type}/{seed}"
    # run_name += f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    output_dir = os.path.join(args.output_dir, args.dataset, run_name)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    
    # --- BENCHMARK CREATION
    image_size = generator_config.model.input_size
    benchmark = get_benchmark(image_size, seed)

    # --- LOGGER CREATION
    loggers = []
    
    loggers.append(TextLogger(open(log_file, "a")))
    loggers.append(CSVLogger(log_dir))

    if args.wandb:
        all_configs = {
            "args": vars(args),
            "generator_config": generator_config,
            "generator_strategy_config": generator_strategy_config,
        }
        if args.solver_type is not None:
            all_configs["solver_config"] = get_configuration(args.solver_config_path)
            all_configs["solver_strategy_config"] = get_configuration(args.solver_strategy_config_path)
        loggers.append(WandBLogger(
            project_name=args.project_name,
            run_name=run_name,
            config=all_configs,
        ))

    checkpoint_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(
            directory=os.path.join(output_dir, "checkpoints"),
        ),
        map_location=device
    )

    # Load checkpoint (if exists in the given storage)
    # If it does not exist, strategy will be None and initial_exp will be 0
    strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()

    if initial_exp > 4:
        print(f"Experiment with seed {seed} already completed")
        return

    if strategy is not None:
        generator_strategy = strategy.generator_strategy
    else:
        # --- STRATEGY CREATION
        generator_strategy = get_generator_strategy(
            args.generator_type,
            generator_config,
            generator_strategy_config,
            loggers,
            device,
            generation_steps=args.generation_steps,
            eta=args.eta,
            checkpoint_plugin=checkpoint_plugin if args.solver_type is None else None,
            lambd=args.lambd,
        )

        if args.solver_type is None:
            strategy = generator_strategy
        else:
            strategy = get_solver_strategy(
                args.solver_type,
                get_configuration(args.solver_config_path),
                get_configuration(args.solver_strategy_config_path),
                generator_strategy,
                loggers,
                device,
                checkpoint_plugin=checkpoint_plugin,
            )

    # TRAINING LOOP
    print("Starting experiment...")
    n_samples = 100
    for experience in benchmark.train_stream[initial_exp:]:
        print("Start of experience ", experience.current_experience)
        strategy.train(experience)
        print("Training completed")

        if args.solver_type is not None:
            print("Computing diffusion metrics on the whole test set")
            generator_strategy.eval(benchmark.test_stream)

        print("Computing accuracy on the whole test set")
        strategy.eval(benchmark.test_stream)

        print("Computing generated samples and saving them to disk")
        evaluate_diffusion(output_dir, n_samples, experience.current_experience,
                           generator_strategy.model, seed=args.seed, generation_steps=args.generation_steps, eta=args.eta)
        
        # if experience.current_experience > 0:
        #     evaluate_diffusion(output_dir + "/old", n_samples, experience.current_experience,
        #                        generator_strategy.old_model, seed=args.seed, generation_steps=args.generation_steps, eta=args.eta)

    print("Evaluation completed")


def main(args):
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    if args.seed  != -1:
        run_experiment(args, args.seed, device)
    else:
        assert not args.wandb, "wandb logging is not supported for multiple seeds"
        for seed in [42, 69, 420, 666, 1714]:
            run_experiment(args, seed, device)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
