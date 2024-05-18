#! /bin/bash
#SBATCH --job-name="neural_2024_agotovtsev"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=0-10:0
#SBATCH --mail-user=artemiygotovtsev@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="out.log"%j.out
#SBATCH --error="err.log"%j.out
#SBATCH --constraint="type_e_debug"
#SBATCH --export=ALL

# Executable
module purge
module load Python

source deactivate
conda activate agotovtsev
nvidia-smi

python3 -u train.py