#!/bin/bash

#SBATCH --job-name=Sobel
#SBATCH --output=Sobel.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH  --gres=gpu:1

imageFolder="../../Images_L7/"
#imageName_band6=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B6_VCID_1.TIF"
imageName_band6=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B6_VCID_2.TIF"

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

i=0
for i in `seq 1 20`;
do
	./build/Thermal_Image.out ${imageName_band6} >> times.txt
done
