#!/bin/bash
#SBATCH --job-name=wgan256
#SBATCH --account=fc_electron
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=andy-wu@berkeley.edu
#SBATCH --output=out/wgan256-%A.out
module load tensorflow/1.12.0-py36-pip-gpu
nvidia-smi
python3 wgan256.py &> 256.out
