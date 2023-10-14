import os
import argparse
import torch
import torch.optim.lr_scheduler

from torchvision import transforms
from diffusers import UNet2DModel
from torch.nn import CrossEntropyLoss

from avalanche.training import Naive, Cumulative, Replay, EWC, SynapticIntelligence, LwF
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
    NaiveDiffusionTraining,
    CumulativeDiffusionTraining,
    EWCDiffusionTraining,
    SIDiffusionTraining,
    VAETraining
)
from src.continual_learning.plugins import UpdatedGenerativeReplayPlugin
from src.continual_learning.metrics.diffusion_metrics import DiffusionMetricsMetric
from src.continual_learning.metrics.loss import loss_metrics, replay_loss_metrics, data_loss_metrics
from src.continual_learning.loggers import TextLogger, CSVLogger
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.schedulers.scheduler_ddim import DDIMScheduler
from src.common.utils import get_configuration
from src.common.diffusion_utils import wrap_in_pipeline, generate_diffusion_samples
from src.models.vae import MlpVAE, VAE_loss
from src.models.simple_cnn import SimpleCNN


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="split_fmnist")
    parser.add_argument("--image_size", type=int, default=32)

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

    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
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
            DiffusionMetricsMetric(device=device),
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
        elif strategy_config.strategy == "naive":
            generator_strategy = NaiveDiffusionTraining(
                generator_model,
                noise_scheduler,
                torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                plugins=plugins,
            )
        elif strategy_config.strategy == "cumulative":
            generator_strategy = CumulativeDiffusionTraining(
                generator_model,
                noise_scheduler,
                torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                plugins=plugins,
            )
        elif strategy_config.strategy == "ewc":
            raise NotImplementedError("EWC is not implemented for diffusion models yet")
            generator_strategy = EWCDiffusionTraining(
                ewc_lambda=strategy_config.ewc_lambda,
                mode=strategy_config.mode,
                decay_factor=strategy_config.decay_factor,
                keep_importance_data=strategy_config.keep_importance_data,
                model=generator_model,
                scheduler=noise_scheduler,
                optimizer=torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                plugins=plugins,
            )
        elif strategy_config.strategy == "si":
            raise NotImplementedError("SI is not implemented for diffusion models yet")
            generator_strategy = SIDiffusionTraining(
                si_lambda=strategy_config.si_lambda,
                eps=strategy_config.eps,
                decay_factor=strategy_config.decay_factor,
                keep_importance_data=strategy_config.keep_importance_data,
                model=generator_model,
                scheduler=noise_scheduler,
                optimizer=torch.optim.Adam(generator_model.parameters(),
                                lr=model_config.optimizer.lr),
                train_mb_size=strategy_config.train_batch_size,
                train_epochs=strategy_config.epochs,
                eval_mb_size=strategy_config.eval_batch_size,
                device=device,
                evaluator=gen_eval_plugin,
                train_timesteps=model_config.scheduler.train_timesteps,
                plugins=plugins,
            )
        else:
            raise NotImplementedError(
                f"Strategy {strategy_config.strategy} not implemented")

    elif generator_type == "vae":
        print("WARNING: VAE code has not been tested in a while...")
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
            DiffusionMetricsMetric(device=device),
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
            trained_experience=True,
        ),
        forgetting_metrics(experience=True, stream=True),
        loggers=loggers,
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    if "strategy" in strategy_config and strategy_config.strategy == "naive":
        cl_strategy = Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=model_config.optimizer.lr),
            CrossEntropyLoss(),
            train_mb_size=strategy_config.train_batch_size,
            train_epochs=strategy_config.epochs,
            eval_mb_size=strategy_config.eval_batch_size,
            device=device,
            evaluator=eval_plugin,
        )
    elif "strategy" in strategy_config and strategy_config.strategy == "cumulative":
        cl_strategy = Cumulative(
            model,
            torch.optim.Adam(model.parameters(), lr=model_config.optimizer.lr),
            CrossEntropyLoss(),
            train_mb_size=strategy_config.train_batch_size,
            train_epochs=strategy_config.epochs,
            eval_mb_size=strategy_config.eval_batch_size,
            device=device,
            evaluator=eval_plugin,
        )
    elif "strategy" in strategy_config and strategy_config.strategy == "er":
        cl_strategy = Replay(
            model,
            torch.optim.Adam(model.parameters(), lr=model_config.optimizer.lr),
            CrossEntropyLoss(),
            train_mb_size=strategy_config.train_batch_size,
            train_epochs=strategy_config.epochs,
            eval_mb_size=strategy_config.eval_batch_size,
            device=device,
            evaluator=eval_plugin,
            mem_size=strategy_config.replay_size,
        )
    elif "strategy" in strategy_config and strategy_config.strategy == "ewc":
        cl_strategy = EWC(
            model,
            torch.optim.Adam(model.parameters(), lr=model_config.optimizer.lr),
            CrossEntropyLoss(),
            ewc_lambda=strategy_config.ewc_lambda,
            mode=strategy_config.mode,
            decay_factor=strategy_config.decay_factor,
            keep_importance_data=strategy_config.keep_importance_data,
            train_mb_size=strategy_config.train_batch_size,
            train_epochs=strategy_config.epochs,
            eval_mb_size=strategy_config.eval_batch_size,
            device=device,
            evaluator=eval_plugin,
        )
    elif "strategy" in strategy_config and strategy_config.strategy == "si":
        cl_strategy = SynapticIntelligence(
            model,
            torch.optim.Adam(model.parameters(), lr=model_config.optimizer.lr),
            CrossEntropyLoss(),
            si_lambda=strategy_config.si_lambda,
            eps=strategy_config.eps,
            train_mb_size=strategy_config.train_batch_size,
            train_epochs=strategy_config.epochs,
            eval_mb_size=strategy_config.eval_batch_size,
            device=device,
            evaluator=eval_plugin,
        )
    elif "strategy" in strategy_config and strategy_config.strategy == "lwf":
        cl_strategy = LwF(
            model,
            torch.optim.Adam(model.parameters(), lr=model_config.optimizer.lr),
            CrossEntropyLoss(),
            alpha=strategy_config.lwf_alpha,
            temperature=strategy_config.lwf_temperature,
            train_mb_size=strategy_config.train_batch_size,
            train_epochs=strategy_config.epochs,
            eval_mb_size=strategy_config.eval_batch_size,
            device=device,
            evaluator=eval_plugin,
        )
    else:
        assert generator_strategy is not None
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
    
    run_name = "gr"

    if args.generator_type is not None and args.generator_type != "None":
        generator_config = get_configuration(args.generator_config_path)
        generator_strategy_config = get_configuration(args.generator_strategy_config_path)
        run_name += f"_{args.generator_type}_{generator_strategy_config.strategy}_steps_{args.generation_steps}_lambd_{args.lambd}"
    else: 
        generator_config = None
        generator_strategy_config = None

    if args.solver_type is not None and args.solver_type != "None":
        solver_config = get_configuration(args.solver_config_path)
        solver_strategy_config = get_configuration(args.solver_strategy_config_path)
        run_name += f"_{args.solver_type}"
        if "strategy" in solver_strategy_config:
            run_name += f"_{solver_strategy_config.strategy}"
    else:
        solver_config = None
        solver_strategy_config = None
    
    run_name += f"/{seed}"
    # run_name += f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    output_dir = os.path.join(args.output_dir, args.dataset, run_name)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")

    if os.path.exists(os.path.join(output_dir, "completed.txt")):
        print(f"Experiment with seed {seed} already completed")
        return
    
    # --- BENCHMARK CREATION
    image_size = args.image_size
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
            "solver_config": solver_config,
            "solver_strategy_config": solver_strategy_config,
        }
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
        if args.generator_type is None or args.generator_type == "None":
            generator_strategy = None  
        else:
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

        if args.solver_type is None or args.solver_type == "None":
            strategy = generator_strategy
        else:
            strategy = get_solver_strategy(
            args.solver_type,
            solver_config,
            solver_strategy_config,
            generator_strategy,
            loggers,
            device,
            checkpoint_plugin=checkpoint_plugin,
        )      

        assert strategy is not None 

    # TRAINING LOOP
    print("Starting experiment...")
    n_samples = 100
    for experience in benchmark.train_stream[initial_exp:]:
        print("Start of experience ", experience.current_experience)
        strategy.train(experience)
        print("Training completed")

        print("Computing metrics on the whole test set")
        if args.solver_type is not None and args.solver_type != "None" and generator_strategy is not None:
            generator_strategy.eval(benchmark.test_stream)

        strategy.eval(benchmark.test_stream)

        if generator_strategy is not None:
            print("Computing generated samples and saving them to disk")
            generate_diffusion_samples(output_dir, n_samples, experience.current_experience,
                            generator_strategy.model, seed=args.seed, generation_steps=args.generation_steps, eta=args.eta)

    print("Evaluation completed")

    with open(os.path.join(output_dir, "completed.txt"), "w") as f:
        f.write(":)")

    # Remove checkpoints
    os.system(f"rm -rf {os.path.join(output_dir, 'checkpoints')}")


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
        for seed in [42, 69, 1714]:
            run_experiment(args, seed, device)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
