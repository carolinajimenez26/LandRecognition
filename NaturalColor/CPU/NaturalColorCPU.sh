#!/bin/bash

#SBATCH --job-name=Sobel
#SBATCH --output=Sobel.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH  --gres=gpu:1

imageFolder="../../Images_L7/"
imageName_band1=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B1.TIF"
imageName_band2=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B2.TIF"
imageName_band3=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B3.TIF"


export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#export CUDA_VISIBLE_DEVICES=0

#./build/SobelFilter.out ../../../images/thor.jpg
i=0

for i in `seq 1 20`;
do
	./build/NaturalColor.out ${imageName_band1} ${imageName_band2} ${imageName_band3} >> times.txt
done
