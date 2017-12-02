#!/bin/bash

#SBATCH --job-name=Sobel
#SBATCH --output=Sobel.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH  --gres=gpu:1

imageFolder="./build/"
#imageName_band1=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B6_VCID_1.TIF"
imageName_band1=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B6_VCID_2.TIF"

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

i=0

for i in `seq 1 20`;
do
	./build/thermal_image_GPU.out ${imageName_band1} >> times.txt
done
