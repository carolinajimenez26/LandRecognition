#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include<opencv2/opencv.hpp>
#include <time.h>
#include <cuda.h>

using namespace std;
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0

#define Channels 3


__global__ void naturalColor(unsigned char *imagen1,unsigned char *imagen2,unsigned char *imagen3, int filas, int columnas, unsigned char *resultado){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < filas) && (col < columnas)){

        int pos = ((row*columnas)+col)*3;
        
        resultado[pos+BLUE] = imagen1[pos+BLUE];
        resultado[pos+GREEN] = imagen2[pos+GREEN];
        resultado[pos+RED] = imagen3[pos+RED];
    }
}


int main(int argc, char **argv){

    if(argc !=4){
        printf("Please, enter the paths of the three images \n");
        return -1;
    }



    cudaError_t error = cudaSuccess;

    clock_t start, end;
    double time_used;
    start = clock();


    //input images host
    unsigned char *h_InputImageB1,*h_InputImageB2,*h_InputImageB3;
    //input images Device
    unsigned char *d_InputImageB1,*d_InputImageB2,*d_InputImageB3;
    //output image Host
    unsigned char *h_ImageResult;
    //output image Device
    unsigned char *d_ImageResult;

    char* imageName1 = argv[1];
    char* imageName2 = argv[2];
    char* imageName3 = argv[3];
    

	Mat image1, image2, image3;


    image1 = imread(imageName1, CV_LOAD_IMAGE_COLOR);
    image2 = imread(imageName2, CV_LOAD_IMAGE_COLOR);
    image3 = imread(imageName3, CV_LOAD_IMAGE_COLOR);    
    
    //se obtienen los atributos de las imagenes

    Size s = image1.size();

    int width = s.width;
    int height = s.height;

    //Se cargan las imagenes

    h_InputImageB1 = image1.data;
    h_InputImageB2 = image2.data;
    h_InputImageB3 = image3.data;



    //reserve memory for Host and device ////////////////////////////////////////
    int size = sizeof(unsigned char)*width*height*image1.channels();


   //se reserva memoria para la imagen de salida en el host 
    h_ImageResult = (unsigned char*)malloc(size);



    //Imagenes de entrada device
    error = cudaMalloc((void**)&d_InputImageB1,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_InputImageB1 \n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_InputImageB2,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_InputImageB2 \n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_InputImageB3,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_InputImageB3 \n");
        exit(-1);
    }

    //Imagen de salida device

     error = cudaMalloc((void**)&d_ImageResult,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_ImageResult \n");
        exit(-1);
    }


    ////////////Copia de imagenes de entrada del host al device/////////////////

    error = cudaMemcpy(d_InputImageB1, h_InputImageB1,size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_InputImageB1 a d_InputImageB1 \n");
        exit(-1);
    }

    //Copia de imagenes de entrada del host al device
    error = cudaMemcpy(d_InputImageB2, h_InputImageB2,size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_InputImageB2 a d_InputImageB2 \n");
        exit(-1);
    }

    //Copia de imagenes de entrada del host al device
    error = cudaMemcpy(d_InputImageB3, h_InputImageB3,size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_InputImageB3 a d_InputImageB3 \n");
        exit(-1);
    }

    //////SE lanza El kernel ////////////

    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);////bloque de 32 x 32 hilos = 1024 hilos
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    naturalColor<<<dimGrid,dimBlock>>>(d_InputImageB1,d_InputImageB2,d_InputImageB3, height, width, d_ImageResult);

    cudaDeviceSynchronize();

    //copian los datos de la imagen del device a la de salida del host
    cudaMemcpy(h_ImageResult,d_ImageResult,size,cudaMemcpyDeviceToHost);


    Mat ImageNaturalColor;
    ImageNaturalColor.create(height,width,CV_8UC3);
    ImageNaturalColor.data = h_ImageResult;

    //imshow("Image Natural color",ImageNaturalColor);
	imwrite("./naturalColor.jpg",ImageNaturalColor);

    //waitKey(0);

    end = clock();
    time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf ("%lf \n",time_used);

    //Se libera memoria
    free(h_ImageResult);

    cudaFree(d_InputImageB1);
    cudaFree(d_InputImageB2);
    cudaFree(d_InputImageB3);

    cudaFree(d_ImageResult);
    
    return 0;
}
