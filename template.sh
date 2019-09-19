#!/bin/bash
# Job name:
#SBATCH --job-name=singleGAN
#
# Account:
#SBATCH --account=fc_electron
#
# Partition:
#SBATCH ==pertition=savio2_gpu
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Bumber o f tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
# Number of GPUs, this can be in the format of "gpu:[1=4]", or "gpu:K80[1-4]" with the type included:
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=02:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andy-wu@berkeley.edu
## Command(s) to run (example):
./a.out
