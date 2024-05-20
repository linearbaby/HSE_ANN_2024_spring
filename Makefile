read_tensorboard:
	scp -r -P 2222 aegotovtsev@cluster.hpc.hse.ru:/home/aegotovtsev/neural_2024/artifacts/result/runs/* tensorboard

move_eval:
	scp -P 2222 eval.py eval.sbatch aegotovtsev@cluster.hpc.hse.ru:/home/aegotovtsev/neural_2024/predictions

move_train:
	scp -P 2222 train.py train.sbatch aegotovtsev@cluster.hpc.hse.ru:/home/aegotovtsev/neural_2024/

ssh_cluster:
	ssh aegotovtsev@cluster.hpc.hse.ru -p 2222