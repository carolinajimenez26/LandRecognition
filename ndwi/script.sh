imageFolder="../../../Documents/Landsat7/images/"
imageName_band4=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B4.TIF"
imageName_band7=$imageFolder"LE07_L1GT_010054_20170421_20170517_01_T2_B7.TIF"

i=0

for i in `seq 1 20`;
do
  echo $i
  ./ndwi.out ${imageName_band4} ${imageName_band7}
done
