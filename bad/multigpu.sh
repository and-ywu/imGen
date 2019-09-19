#!/bin/bash
#SBATCH --job-name=wganmulti
#SBATCH --account=fc_electron
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=andy-wu@berkeley.edu
#SBATCH --output=out/wganmulti-%A.out
module load tensorflow/1.12.0-py36-pip-gpu
nvidia-smi
python3 wganmulti.py
