#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include<opencv2/opencv.hpp>
#include <math.h>
#include <time.h>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

#define dbg(x) cout << #x << ": " << x << endl

#define RADIANCE_MULT_BAND_7 0.043898
#define RADIANCE_ADD_BAND_7 -0.39390

#define RADIANCE_MULT_BAND_4 0.96929
#define RADIANCE_ADD_BAND_4 -6.06929

#define REFLECTANCE_MULT_BAND_7 0.0017122
#define REFLECTANCE_ADD_BAND_7 -0.015363

#define REFLECTANCE_MULT_BAND_4 0.0028720
#define REFLECTANCE_ADD_BAND_4 -0.017983

#define theta_SE 65.46088663

#define BLUE 0
#define GREEN 1
#define RED 2


void saveTime(double elapsedTime, string fileName){
	FILE *stream;
	stream = fopen("ndwi_time.txt", "a");
	fprintf(stream, "%f\n", elapsedTime);
	fclose(stream);
}

double getRadiance(int grayPixel, int band){
	if (grayPixel == 0)
		return 0;

	if (band == 4)
		return grayPixel * RADIANCE_MULT_BAND_4 + RADIANCE_ADD_BAND_4;
	return grayPixel * RADIANCE_MULT_BAND_7 + RADIANCE_ADD_BAND_7;
}

double getReflectance(int pixel, int band){
	if (pixel == 0)
		return 0;

	double theta = (theta_SE * M_PI) / 180;
	if (band == 4)
		return (REFLECTANCE_MULT_BAND_4 * pixel + REFLECTANCE_ADD_BAND_4) / sin(theta);
	return (REFLECTANCE_MULT_BAND_7 * pixel + REFLECTANCE_ADD_BAND_7) / sin(theta);
}

void setRadiance(unsigned char *data, int const &height, int const &width, int const &band){

	double value = 0.0;
	int pos;
	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			pos = row*width+col;
			value = getRadiance(data[pos], band);
			data[pos] = value;

		}
	}
}

void setReflectance(unsigned char *data, int const &height, int const &width, int const &band){

	double value = 0.0;
	int pos;
	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			pos = row*width+col;
			value = getReflectance((int)data[pos], band);
			data[pos] = (double)value;

		}
	}
}

void setHighBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 193;
	image[pos + GREEN] = 134;
	image[pos + RED] = 46;
}

void setMediumBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 226;
	image[pos + GREEN] = 173;
	image[pos + RED] = 93;
}

void setLowBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 241;
	image[pos + GREEN] = 214;
	image[pos + RED] = 174;
}

void setHigherBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 140;
	image[pos + GREEN] = 97;
	image[pos + RED] = 33;
}

void setMoreHigherBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 114;
	image[pos + GREEN] = 79;
	image[pos + RED] = 27;
}

void setLowYellow(unsigned char *image, int pos){
	image[pos + BLUE] = 161;
	image[pos + GREEN] = 255;
	image[pos + RED] = 246;
}

void setMediumYellow(unsigned char *image, int pos){
	image[pos + BLUE] = 74;
	image[pos + GREEN] = 255;
	image[pos + RED] = 239;
}

void setHighYellow(unsigned char *image, int pos){
	image[pos + BLUE] = 0;
	image[pos + GREEN] = 245;
	image[pos + RED] = 222;
}

void setBlack(unsigned char *image, int pos){
	image[pos + BLUE] = 0;
	image[pos + GREEN] = 0;
	image[pos + RED] = 0;
}

void setGreen(unsigned char *image, int pos){
	image[pos + BLUE] = 113;
	image[pos + GREEN] = 204;
	image[pos + RED] = 46;
}

void setRed(unsigned char *image, int pos){
	image[pos + BLUE] = 99;
	image[pos + GREEN] = 112;
	image[pos + RED] = 236;
}

void normalize(double const &min, double const &max, double *image, int const &height, int const &width, unsigned char *out){

	int posGray, posResult;
	double value = 0.0;
	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			posGray = (row*width)+col;
			posResult = ((row*width)+col)*3;
			value = image[posGray];

			// if (value >= 0.2 && value <= 1)
			// 	setMoreHigherBlue(out, posResult);
			// if (value > 0 && value < 0.2)
			// 	setMediumBlue(out, posResult);
			// if (value >= -0.184313726 && value < 0)
			// 	setGreen(out, posResult);
			// if (value >= -0.247058824 && value < -0.184313726)
			// 	setMediumYellow(out, posResult);
			// if (value >= -1 && value < -0.247058824)
			// 	setRed(out, posResult);
			// /*if (value == 0)
			// 	setBlack(out, posResult);*/


			if (value > 0.104834912 && value <= 1)
				setLowBlue(out, posResult);
			else if (value > -0.0350346633 && value <= 0.104834912)
				setMediumBlue(out, posResult);
			else if (value > -0.158119833 && value <= -0.0350346633)
				setHighBlue(out, posResult);
			else if (value > -0.230851996 && value <= -0.158119833)
				setHigherBlue(out, posResult);
			else if (value > -0.426669359 && value <= -0.230851996)
				setMoreHigherBlue(out, posResult);
			if (value == 0)
				setBlack(out, posResult);
			
		}
	}
}

void setNDVI(unsigned char *NIR, unsigned char *shortWave, unsigned char *result, int const &height, int const &width){

	int posGray;
	double minValue = -1.0, maxValue = 1.0;
	double value = 0.0;
	int size = sizeof(double)*width*height;
	double *temp;
	temp = (double*)malloc(size);

	int i, posResult;
	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			posGray = (row*width)+col;
			double aux1 = getReflectance(NIR[posGray], 4);
			double aux2 = getReflectance(shortWave[posGray], 5);

			if (aux1 + aux2 != 0){
				value = (aux1 - aux2) / (aux1 + aux2);
				temp[posGray] = value;
				//cout << "NDVI: " <<(double)value << endl;
			}else {
				temp[posGray] = shortWave[posGray];
				//cout << "RED: " <<(double)redVisible[posGray] << endl;
			}

		}
	}

	normalize(-1, 1, temp, height, width, result);
	free(temp);

}

int main(int argc, char **argv)
{
	if (argc != 3){
		cout << "Missing parameters" << endl;
		exit(1);
	}

	clock_t start, end;
	double time_used;
	start = clock();

	unsigned char *shortWave;
	unsigned char *NIR;
	char* NIRPath = argv[1]; //band 4
	char* shortWavePath = argv[2]; //band 5

	Mat shortWaveMat, NIRMat;
	NIRMat = imread(NIRPath, 0);
	shortWaveMat = imread(shortWavePath, 0);
    Size s = NIRMat.size();
    int width = s.width;
    int height = s.height;

    if (!shortWaveMat.data || !NIRMat.data){
    	cout << "error loading image" << endl;
    	exit(1);
    }


    shortWave = shortWaveMat.data;
    NIR = NIRMat.data;

    setRadiance(NIR, height, width, 4);
    setRadiance(shortWave, height, width, 7);

    //--------------------- for debug -------------
    Mat NIRRadiance;
    NIRRadiance.create(height,width,CV_8UC1);
    NIRRadiance.data = NIR;
    imwrite("./NIRRadiance.png", NIRRadiance);

    Mat shortWaveRadiance;
    shortWaveRadiance.create(height,width,CV_8UC1);
    shortWaveRadiance.data = shortWave;
    imwrite("./shortWaveradiance.png", shortWaveRadiance);

    //------------------------------------------------

    int size = sizeof(unsigned char)*width*height*3;
    unsigned char* result;
    result = (unsigned char*)malloc(size);

    setNDVI(NIR, shortWave, result, height, width);
    Mat imageResult;
    imageResult.create(height,width,CV_8UC3);
    imageResult.data = result;

    imwrite("./NDWIimage.jpg", imageResult);

		end = clock();

		time_used = ((double) (end - start))/CLOCKS_PER_SEC;
		saveTime(time_used, "ndvi_time.txt");

    //Se libera memoria
    free(result);

	return 0;
}
