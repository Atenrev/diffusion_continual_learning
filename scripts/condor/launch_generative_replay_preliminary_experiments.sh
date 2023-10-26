condor_send -c 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA} python train_cl.py --generator_strategy_config_path "configs/strategy/diffusion_no_distill_preliminary_20.json" --generation_steps 20 --seed -1 --output_dir "results_fuji/smasipca/generative_replay_preliminary_acc/" --solver_strategy_config_path "configs/strategy/cnn_w_diffusion.json"'
condor_send -c 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA} python train_cl.py --generator_strategy_config_path "configs/strategy/diffusion_no_distill_preliminary_10.json" --generation_steps 10 --seed -1 --output_dir "results_fuji/smasipca/generative_replay_preliminary_acc/" --solver_strategy_config_path "configs/strategy/cnn_w_diffusion.json"'
condor_send -c 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA} python train_cl.py --generator_strategy_config_path "configs/strategy/diffusion_no_distill_preliminary_5.json" --generation_steps 5 --seed -1 --output_dir "results_fuji/smasipca/generative_replay_preliminary_acc/" --solver_strategy_config_path "configs/strategy/cnn_w_diffusion.json"'
condor_send -c 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA} python train_cl.py --generator_strategy_config_path "configs/strategy/diffusion_no_distill_preliminary_2.json" --generation_steps 2 --seed -1 --output_dir "results_fuji/smasipca/generative_replay_preliminary_acc/" --solver_strategy_config_path "configs/strategy/cnn_w_diffusion.json"'