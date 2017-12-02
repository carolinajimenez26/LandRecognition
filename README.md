# LandRecognition

Land recognition by multispectral images from Landsat satelite.
Multispectral images were processed for get Natural color image, thermic image, NDVI image and NDWI image.

Also the images were processed sequentially and in parallel  way, with CUDA in GPU, for compare the results.

### Images

The images were downloaded from https://glovis.usgs.gov/


### Compile and run

Create Makefile with cmake, each folder has its own CMakeLists.txt to create the Makefile
Makefile is runned with "make" command.
