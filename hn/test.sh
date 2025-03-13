#!/bin/sh

#PBS -q a6500g10q
#PBS -l select=1:ngpus=1:ncpus=1:mem=1000m
#PBS -l walltime=00:01:00
#PBS -j oe

cd $PBS_O_WORKDIR
/usr/bin/nvidia-smi -L
echo
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
make

