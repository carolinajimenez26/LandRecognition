#!/bin/bash

#SBATCH --job-name=ndvi
#SBATCH --output=ndvi_result.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH  --gres=gpu:1


export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

imageFolder="../../../Documents/Landsat7/images/"
imageName_band4=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B4.TIF"
imageName_band7=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B7.TIF"

i=0

for i in `seq 1 20`;
do
  echo $i
  ./ndwi.out ${imageName_band4} ${imageName_band7}
done
