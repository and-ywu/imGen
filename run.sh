#!/bin/bash
#SBATCH --job-name=wgan
#SBATCH --account=fc_electron
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=andy-wu@berkeley.edu
#SBATCH --output=out/wgan-%A.out
module load tensorflow/1.12.0-py36-pip-gpu
nvidia-smi
python3 wgan.py
