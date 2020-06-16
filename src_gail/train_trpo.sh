# CUDA_VISIBLE_DEVICES=9 mpirun -np 10 python3 trpo_torque.py --config config/trpo_biped_torque.yaml
CUDA_VISIBLE_DEVICES=8 mpirun -np 10 python3 trpo_PID.py --config config/trpo_biped_PID.yaml
