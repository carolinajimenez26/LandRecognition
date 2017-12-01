#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include<opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

#define RADIANCE_MULT_BAND_6_VCID_1  0.067087
#define RADIANCE_ADD_BAND_6_VCID_1  -0.06709
#define K1_CONSTANT_BAND_6_VCID_1  666.09
#define K2_CONSTANT_BAND_6_VCID_1  1282.71

int clamp(int const &pixel){
    if (pixel > 255)
        return 255;
    else if (pixel < 0)
        return 0;
    else return pixel;
}

double getRadiance(int grayPixel){
	return grayPixel * RADIANCE_MULT_BAND_6_VCID_1 + RADIANCE_ADD_BAND_6_VCID_1;
}

double getBrightness(int pixel){
    return (K2_CONSTANT_BAND_6_VCID_1 / log((K1_CONSTANT_BAND_6_VCID_1 / pixel) + 1) ) - 272.15;
}

void thermicImage(unsigned char *imagen, int const &height, int const &width, unsigned char *resultado){
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			int pos = ((i*width)+j)*3;
			resultado[pos] = clamp(getRadiance(imagen[pos]));
            resultado[pos] = clamp(getBrightness(resultado[pos]));
		}
	}
}

int main(int argc, char **argv)
{
	unsigned char *InputImageB;
	unsigned char *ImageResult;

	if(argc !=2){
        printf("Please, enter the paths of the thermic image \n");
        exit(1);
    }

    char* imageName = argv[1];

    Mat image;
    image = imread(imageName, CV_LOAD_IMAGE_COLOR);
    Size s = image.size();
    int width = s.width;
    int height = s.height;

    InputImageB = image.data;

    int size = sizeof(unsigned char)*width*height*image.channels();
    //se reserva memoria para la imagen de salida  
    ImageResult = (unsigned char*)malloc(size);

    thermicImage(InputImageB, height, width, ImageResult);

    Mat ImageThermic;
    ImageThermic.create(height,width,CV_8UC3);
    ImageThermic.data = ImageResult;

    imwrite("./thermicImage.jpg",ImageThermic);

    waitKey(0);

    //Se libera memoria
    free(ImageResult);

	return 0;
}