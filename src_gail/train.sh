# CUDA_VISIBLE_DEVICES=9 mpirun -np 10 python3 trpo_torque.py --config config/trpo_biped_torque.yaml
# CUDA_VISIBLE_DEVICES=8 mpirun -np 10 python3 trpo_PID.py --config config/trpo_biped_PID.yaml
# CUDA_VISIBLE_DEVICES=8 mpirun -np 10 python3 gail_ppo_PID_norm2.py --config config/gail_ppo_biped_PID.yaml
# CUDA_VISIBLE_DEVICES=8 mpirun -np 10 python3 gail_ppo_PID_norm3.py --config config/gail_ppo_biped_PID.yaml
# CUDA_VISIBLE_DEVICES=2 mpirun -np 10 python3 gail_ppo_PID_norm3.py --config config/gail_ppo_biped_PID.yaml
# CUDA_VISIBLE_DEVICES=7 mpirun -np 10 python3 gail_ppo_PID_unnorm_speed.py --config config/gail_ppo_biped_PID_unnorm_speed.yaml
# CUDA_VISIBLE_DEVICES=9 mpirun -np 10 python3 gail_ppo_PID_multiClip_GP.py --config config/gail_ppo_biped_PID_multiClip.yaml
CUDA_VISIBLE_DEVICES=7 mpirun -np 10 python3 gail_ppo_PID_unnorm_variable_speed_graph.py --config config/gail_ppo_PID_unnorm_variable_speed_graph.yaml
