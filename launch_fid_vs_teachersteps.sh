condor_send -c 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA} python fid_vs_teachersteps.py --distillation_type generation'
condor_send -c 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA} python fid_vs_teachersteps.py --distillation_type partial_generation'
condor_send -c 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA} python fid_vs_teachersteps.py --distillation_type no_distillation'