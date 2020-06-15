# CUDA_VISIBLE_DEVICES=8 mpirun -np 10 python3 gail_trpo_torque.py --config config/gail_trpo_biped_torque.yaml
CUDA_VISIBLE_DEVICES=8 mpirun -np 10 python3 gail_ppo_PID.py --config config/gail_ppo_biped_PID.yaml
