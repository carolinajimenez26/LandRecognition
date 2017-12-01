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

__device__
double getRadiance(int grayPixel, int band){
	if (grayPixel == 0)
		return 0;

	if (band == 4)
		return grayPixel * RADIANCE_MULT_BAND_4 + RADIANCE_ADD_BAND_4;
	return grayPixel * RADIANCE_MULT_BAND_7 + RADIANCE_ADD_BAND_7;
}

__device__
double getReflectance(int pixel, int band){
	if (pixel == 0)
		return 0;

	double theta = (theta_SE * M_PI) / 180;
	if (band == 4)
		return (REFLECTANCE_MULT_BAND_4 * pixel + REFLECTANCE_ADD_BAND_4) / sin(theta);
	return (REFLECTANCE_MULT_BAND_7 * pixel + REFLECTANCE_ADD_BAND_7) / sin(theta);
}

__global__
void setRadiance(unsigned char *data, int height, int width, int band){

	double value = 0.0;
	int pos;
	int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;

	if (i < height and j < width) {
			pos = i * width + j;
			value = getRadiance(data[pos], band);
			data[pos] = value;
	}
}

__device__
void setHighBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 193;
	image[pos + GREEN] = 134;
	image[pos + RED] = 46;
}

__device__
void setMediumBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 226;
	image[pos + GREEN] = 173;
	image[pos + RED] = 93;
}

__device__
void setLowBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 241;
	image[pos + GREEN] = 214;
	image[pos + RED] = 174;
}

__device__
void setHigherBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 140;
	image[pos + GREEN] = 97;
	image[pos + RED] = 33;
}

__device__
void setMoreHigherBlue(unsigned char *image, int pos){
	image[pos + BLUE] = 114;
	image[pos + GREEN] = 79;
	image[pos + RED] = 27;
}

__device__
void setLowYellow(unsigned char *image, int pos){
	image[pos + BLUE] = 161;
	image[pos + GREEN] = 255;
	image[pos + RED] = 246;
}

__device__
void setMediumYellow(unsigned char *image, int pos){
	image[pos + BLUE] = 74;
	image[pos + GREEN] = 255;
	image[pos + RED] = 239;
}

__device__
void setHighYellow(unsigned char *image, int pos){
	image[pos + BLUE] = 0;
	image[pos + GREEN] = 245;
	image[pos + RED] = 222;
}

__device__
void setBlack(unsigned char *image, int pos){
	image[pos + BLUE] = 0;
	image[pos + GREEN] = 0;
	image[pos + RED] = 0;
}

__device__
void setGreen(unsigned char *image, int pos){
	image[pos + BLUE] = 113;
	image[pos + GREEN] = 204;
	image[pos + RED] = 46;
}

__device__
void setRed(unsigned char *image, int pos){
	image[pos + BLUE] = 99;
	image[pos + GREEN] = 112;
	image[pos + RED] = 236;
}

__device__
void normalize(double min, double max, double *image, int height, int width,
							 unsigned char *out){

	int posGray, posResult;
	double value = 0.0;

	int i = blockIdx.y*blockDim.y+threadIdx.y;
	int j = blockIdx.x*blockDim.x+threadIdx.x;

	if (i < height and j < width) {
			posGray = i * width + j;
			posResult = (i * width + j)*3;
			value = image[posGray];

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

__global__
void setNDVI(unsigned char *NIR, unsigned char *shortWave, unsigned char *result,
						 int height, int width, double *temp){

	int posGray;
	double minValue = -1.0, maxValue = 1.0;
	double value = 0.0;

	int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;

	int posResult;
	if (i < height and j < width) {
			posGray = i * width + j;
			double aux1 = getReflectance(NIR[posGray], 4);
			double aux2 = getReflectance(shortWave[posGray], 5);

			if (aux1 + aux2 != 0){
				value = (aux1 - aux2) / (aux1 + aux2);
				temp[posGray] = value;
				//cout << "NDVI: " <<(double)value << endl;
			} else {
				temp[posGray] = shortWave[posGray];
				//cout << "RED: " <<(double)redVisible[posGray] << endl;
			}

	}

	__syncthreads();
	normalize(-1, 1, temp, height, width, result);
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

	cudaError_t error = cudaSuccess;
	unsigned char *shortWave, *NIR, *d_shortWave, *d_NIR;
	char* NIRPath = argv[1]; //band 4
	char* shortWavePath = argv[2]; //band 5

	Mat shortWaveMat, NIRMat;
	NIRMat = imread(NIRPath, 0);
	shortWaveMat = imread(shortWavePath, 0);

  if (!shortWaveMat.data || !NIRMat.data){
  	cout << "error loading image" << endl;
  	exit(1);
  }

	Size s = NIRMat.size();
	int width = s.width;
	int height = s.height;

	shortWave = (unsigned char*)malloc(height * width * sizeof(unsigned char));
	NIR = (unsigned char*)malloc(height * width * sizeof(unsigned char));

  shortWave = shortWaveMat.data;
  NIR = NIRMat.data;

	error = cudaMalloc((void**)&d_shortWave, width * height * sizeof(unsigned char));
  if (error != cudaSuccess) {
  	printf("Error allocating memory to d_shortWave");
    return 1;
	}

	error = cudaMalloc((void**)&d_NIR, width * height * sizeof(unsigned char));
  if (error != cudaSuccess) {
  	printf("Error allocating memory to d_NIR");
    return 1;
	}

	error = cudaMemcpy(d_shortWave, shortWave, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("Error copying memory to d_shortWave");
		return 1;
	}

	error = cudaMemcpy(d_NIR, NIR, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("Error copying memory to d_NIR");
		return 1;
	}

	int blockSize = 32;
  dim3 dimblock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil((width) / float(blockSize)), ceil((height) / float(blockSize)), 1);

  setRadiance<<<dimGrid,dimblock>>>(d_NIR, height, width, 4);
  cudaDeviceSynchronize();

	error = cudaMemcpy(NIR, d_NIR, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
  	printf("Error copying memory to NIR");
    return 1;
	}

	setRadiance<<<dimGrid,dimblock>>>(d_shortWave, height, width, 7);
  cudaDeviceSynchronize();

	error = cudaMemcpy(shortWave, d_shortWave, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
  	printf("Error copying memory to shortWave");
    return 1;
	}

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

  int size = sizeof(unsigned char) * width * height * 3;
  unsigned char *result, *d_result;
	double *d_temp;

  result = (unsigned char*)malloc(size);

	error = cudaMalloc((void**)&d_result, size);
  if (error != cudaSuccess) {
  	printf("Error allocating memory to d_result");
    return 1;
	}

	error = cudaMalloc((void**)&d_temp, sizeof(double) * width * height);
  if (error != cudaSuccess) {
  	printf("Error allocating memory to d_temp");
    return 1;
	}

	setNDVI<<<dimGrid,dimblock>>>(d_NIR, d_shortWave, d_result, height, width, d_temp);
	cudaDeviceSynchronize();

	error = cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
  	printf("Error copying memory to result");
    return 1;
	}

  Mat imageResult;
  imageResult.create(height,width,CV_8UC3);
  imageResult.data = result;

  imwrite("./NDWIimage.jpg", imageResult);

	end = clock();

	time_used = ((double) (end - start))/CLOCKS_PER_SEC;
	saveTime(time_used, "ndvi_time.txt");

  //Se libera memoria
	// free(shortWave); free(redVisible);
	free(result);
  cudaFree(d_result);
  cudaFree(d_temp);
	cudaFree(d_shortWave); cudaFree(d_NIR);

	return 0;
}
