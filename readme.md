# Continual Learning of Diffusion Models with Generative Distillation

A PyTorch implementation of the continual learning experiments with Diffusion Models described in the following paper:

- Continual Learning of Diffusion Models with Generative Distillation: https://arxiv.org/abs/2311.14028

This paper proposes to introduce knowledge distillation into generative replay of diffusion models, which substantially improves the performance of the continually trained model.


## Installation and requirements

This code makes use of the following libraries:
- Python 3.9.16
- PyTorch 2.0.1
- Diffusers 0.16.1
- avalanche-lib 0.3.1
- torch-fidelity 0.3.0

To use the code, first clone the repository:

```bash
git clone https://github.com/Atenrev/difussion_continual_learning.git
cd difussion_continual_learning
```

Then, assuming you have Python 3.9.16 set up, install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the code

The code allows to run the following experiments:

- **IID experiments**: train a diffusion model on an Independent and Identically Distributed (IID) dataset, and evaluate it on a test set. These experiments allow to set an upper target on the performance of the alternative distillation methods.
- **Continual learning experiments**: train a diffusion model on a continual learning scenario, and evaluate it on a test set. These experiments are used to evaluate the effectivenes of the alternative distillation methods in a continual learning scenario. To replicate the experiments in the paper, please refer to this section.

The script for both types of experiments expect a configuration file for the model. The configuration files for the models used in the paper are in the ``configs/model`` folder. We used ``ddim_medium.json`` consistently throughout our experiments. The configuration files for the continual learning strategies are in the ``configs/strategy`` folder. You can also write your own configuration files, and use them with the scripts. If so, stick to the same format as the configuration files in the ``configs`` folder. 

### IID experiments

To run the IID experiments, use the following command:

```bash
train_iid.py [-h] [--image_size IMAGE_SIZE] [--channels CHANNELS] [--dataset DATASET] [--model_config_path MODEL_CONFIG_PATH] [--training_type TRAINING_TYPE]
                    [--distillation_type DISTILLATION_TYPE] [--teacher_path TEACHER_PATH] [--criterion CRITERION] [--generation_steps GENERATION_STEPS] [--eta ETA]
                    [--teacher_generation_steps TEACHER_GENERATION_STEPS] [--teacher_eta TEACHER_ETA] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE]
                    [--results_folder RESULTS_FOLDER] [--save_every SAVE_EVERY] [--use_wandb] [--seed SEED]  
```

where the arguments are:

- ``image_size``: Size of images to use for training
- ``channels``: Number of channels to use for training
- ``dataset``: Dataset to use for training (mnist, fashion_mnist, cifar100)
- ``model_config_path``: Path to model configuration file
- ``training_type``: Type of training to use (evaluate, diffusion, generative)
- ``distillation_type``: Type of distillation to use (gaussian, gaussian_symmetry, generation, partial_generation, no_distillation)
- ``teacher_path``: Path to teacher model (only for distillation)
- ``criterion``: Criterion to use for training (smooth_l1, mse, min_snr)
- ``generation_steps``: Number of steps for diffusion (used in evaluation)
- ``eta``: Eta for diffusion (used in evaluation)
- ``teacher_generation_steps``: Number of steps for teacher diffusion (used in distillation)
- ``teacher_eta``: Eta for teacher diffusion (used in distillation)
- ``num_epochs``: Number of epochs (when not using distillation) or iterations (when using distillation) to train for
- ``batch_size``: Batch size to use for training
- ``eval_batch_size``: Batch size to use for evaluation
- ``results_folder``: Folder to save results to
- ``save_every``: Evaluate and save model every n epochs (normal) or n iterations (distillation)
- ``use_wandb``: Whether to use wandb for logging
- ``seed``: Seed to use for training. If None, train with 5 different seeds and report the best one


For more information, please use the ``-h`` flag.

Before training a model using distillation, you need to train a teacher model. To train a teacher model on Fashion-MNIST, use the following command:

```bash
python train_iid.py --dataset "fashion_mnist" --model_config_path "configs/model/ddim_medium.json" --num_epochs 100 --results_folder "results/iid/" --seed 42
```

This will train a diffusion model with the configuration in ``configs/model/ddim_medium.json`` for 100 epochs on Fashion-MNIST, and save the model to ``results/iid/fashion_mnist/diffusion/None/ddim_medium_mse/``. The model will be trained with seed 42, and the results will be saved to ``results/iid/fashion_mnist/diffusion/None/ddim_medium_mse/42``. To access the model saved after the last epoch, use the path ``results/iid/fashion_mnist/diffusion/None/ddim_medium_mse/42/last_model``.

For example, to train a diffusion model using the generative distillation method with the teacher you trained with the previous command, use the following command:

```bash
python train_iid.py --model_config_path "configs/model/ddim_medium.json" --distillation_type generation --save_every 1000 --num_epochs 20000 --teacher_generation_steps 2 --teacher_eta 0.0 --teacher_path "results/iid/fashion_mnist/diffusion/None/ddim_medium_mse/42/last_model" --results_folder "results/iid/"
```


### Continual learning experiments

To run the continual learning experiments, use the following command:

```bash
train_cl.py [-h] [--dataset {split_fmnist,split_mnist}] [--image_size IMAGE_SIZE] [--generator_type {diffusion,vae,None}] [--generator_config_path GENERATOR_CONFIG_PATH]
                   [--generator_strategy_config_path GENERATOR_STRATEGY_CONFIG_PATH] [--lambd LAMBD] [--generation_steps GENERATION_STEPS] [--eta ETA] [--solver_type {mlp,cnn,None}]
                   [--solver_config_path SOLVER_CONFIG_PATH] [--solver_strategy_config_path SOLVER_STRATEGY_CONFIG_PATH] [--seed SEED] [--cuda CUDA] [--output_dir OUTPUT_DIR]
                   [--project_name PROJECT_NAME] [--wandb]
```

where the arguments are:

- ``dataset``: Dataset to use for the benchmark (split_fmnist, split_mnist)
- ``image_size``: Image size to use for the benchmark
- ``generator_type``: Type of generator to use for generative replay (diffusion, vae, None)
- ``generator_config_path``: Path to the configuration file of the generator
- ``generator_strategy_config_path``: Path to the configuration file of the generator strategy
- ``lambd``: Lambda parameter used in the generative replay loss of the generator
- ``generation_steps``: Number of steps to use for the diffusion process in evaluation and generative replay of the classifier
- ``eta``: Eta parameter used in the generative replay loss of the generator
- ``solver_type``: Type of solver to use for the benchmark (mlp, cnn, None)
- ``solver_config_path``: Path to the configuration file of the solver
- ``solver_strategy_config_path``: Path to the configuration file of the solver strategy
- ``seed``: Seed to use for the experiment. -1 to run the experiment with seeds 42, 69, 1714
- ``cuda``: Select zero-indexed cuda device. -1 to use CPU.
- ``output_dir``: Output directory for the results
- ``project_name``: Name of the wandb project
- ``wandb``: Use wandb for logging

For more information, please use the ``-h`` flag.

For example, to run the continual learning experiment on Split Fashion-MNIST with a diffusion generator and a CNN solver using the generative distillation strategy, use the following command:

```bash
python train_cl.py --generator_strategy_config_path "configs/strategy/diffusion_full_gen_distill.json" --generation_steps 10 --lambd 3.0 --seed -1 --output_dir results/continual_learning/ --solver_strategy_config_path "configs/strategy/cnn_w_diffusion.json"
```

This will run the experiment with seeds 42, 69 and 1714, and save the results to ``results/continual_learning/dataset_name``. In this case, the results will be saved to ``results/continual_learning/split_fmnist/gr_diffusion_full_generation_distillation_steps_10_lambd_3.0_cnn``. Inside this folder, you will find a folder for each seed, and inside each of these folders you will find the logs in CSV format inside the ``logs`` folder.


## Visualizing the results

You can generate plots of the results using the ``generate_report_iid.py`` and ``generate_report_cl.py`` scripts in the ``utils`` folder. These scripts expect a folder with the results folder of one or more experiments, and generate a report with the results. For example, to generate a report for the IID experiments, use the following command:

```bash
python utils/generate_report_iid.py --experiments_path results/iid/fashion_mnist/diffusion/
```

This will generate a report with the results of the experiments in the ``results/iid/diffusion/`` folder. 

Similarly, to generate a report for the continual learning experiments, use the following command:

```bash
python utils/generate_report_cl.py --experiments_path results/continual_learning/split_fmnist/
```


## Citation

If you use this code in your research, please cite the following paper:

```
@article{masip2023continual,
   title={Continual Learning of Diffusion Models with Generative Distillation},
   author={Sergi Masip and Pau Rodriguez and Tinne Tuytelaars and Gido M. van de Ven},
   journal={arXiv preprint arXiv:2311.14028},
   year={2023}
 }
```
