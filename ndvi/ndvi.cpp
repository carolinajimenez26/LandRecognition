#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include<opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

#define RADIANCE_MULT_BAND_3 0.62165
#define RADIANCE_ADD_BAND_3 -5.62165

#define RADIANCE_MULT_BAND_5 0.12622
#define RADIANCE_ADD_BAND_5 -1.12622

#define REFLECTANCE_MULT_BAND_3 0.0012936
#define REFLECTANCE_ADD_BAND_3 -0.011698

#define REFLECTANCE_MULT_BAND_5 0.0018075
#define REFLECTANCE_ADD_BAND_5 -0.016128

#define theta_SE 65.46088663

double clamp(int const &pixel){
    if (pixel > 255.0)
        return 255.0;
    else if (pixel < 0.0)
        return 0.0;
    else return pixel;
}

double getRadiance(int grayPixel, int band){
	if (band == 3)
		return grayPixel * RADIANCE_MULT_BAND_3 + RADIANCE_ADD_BAND_3;
	return clamp(grayPixel * RADIANCE_MULT_BAND_5 + RADIANCE_ADD_BAND_5);
}

double getReflectance(int pixel, int band){
	if (band == 3)
		return (REFLECTANCE_MULT_BAND_3 * pixel + REFLECTANCE_ADD_BAND_3) / sin(theta_SE);
	return clamp((REFLECTANCE_MULT_BAND_5 * pixel + REFLECTANCE_ADD_BAND_5) / sin(theta_SE));
}

void setRadiance(unsigned char *data, int const &height, int const &width, int const &band){

	double value = 0.0;
	for (int i = 0; i < height*width; i++){
		value = getRadiance(data[i], band);
		data[i] = value;
	}
}

void setReflectance(unsigned char *data, int const &height, int const &width, int const &band){
	
	double value = 0.0;
	for (int i = 0; i < height*width; i++){
		value = getReflectance(data[i], band);
		data[i] = value;
	}
}

void setNDVI(unsigned char *shortWave, unsigned char *redVisible, unsigned char *result, int const &height, int const &width){

	double value = 0.0;
	for (int i = 0; i < height*width; i++){
		if (shortWave[i] + redVisible[i] != 0){
			value = (shortWave[i] - redVisible[i]) / (shortWave[i] + redVisible[i]);
			result[i] = clamp(value);
		}
	}
}

int main(int argc, char **argv)
{
	if (argc != 3){
		cout << "Missing parameters" << endl;
		exit(1);
	}

	unsigned char *shortWave;
	unsigned char *redVisible;
	char* redVisiblePath = argv[1];
	char* shortWavePath = argv[2];

	Mat shortWaveMat, redVisibleMat;
	redVisibleMat = imread(redVisiblePath, CV_LOAD_IMAGE_COLOR);
	shortWaveMat = imread(shortWavePath, CV_LOAD_IMAGE_COLOR);
    Size s = redVisibleMat.size();
    int width = s.width;
    int height = s.height;

    shortWave = shortWaveMat.data;
    redVisible = redVisibleMat.data;

    cout << "DEBUG A" << endl;
    setRadiance(redVisible, height, width, 3);
    cout << "DEBUG B" << endl;
    setRadiance(shortWave, height, width, 4);
    cout << "DEBUG C" << endl;
    setReflectance(redVisible, height, width, 3);
    cout << "DEBUG D" << endl;
    setReflectance(shortWave, height, width, 4);
    cout << "DEBUG E" << endl;
    int size = sizeof(unsigned char)*width*height*3;
    unsigned char* result;
    result = (unsigned char*)malloc(size); 

    setNDVI(shortWave, redVisible, result, height, width);
    cout << "DEBUG F" << endl;
    Mat imageResult;
    imageResult.create(height,width,CV_8UC3);
    imageResult.data = result;

    imwrite("./NDVIimage.jpg", imageResult);

    waitKey(0);

    //Se libera memoria
    free(result);

	return 0;
}