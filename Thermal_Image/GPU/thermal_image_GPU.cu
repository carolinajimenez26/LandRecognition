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

#define ganancia 0.037205 //Grescale Gain ganancia
#define offset 3.16280   //Brescale Bias

#define K1 666.09
#define K2 1282.71

//RADIANCE_MULT_BAND_6_VCID_1 = 6.7087E-02
//RADIANCE_MULT_BAND_6_VCID_2 = 3.7205E-02

//RADIANCE_ADD_BAND_6_VCID_1 = -0.06709
//RADIANCE_ADD_BAND_6_VCID_2 = 3.16280

// K1_CONSTANT_BAND_6_VCID_1 = 666.09
// K2_CONSTANT_BAND_6_VCID_1 = 1282.71

// K1_CONSTANT_BAND_6_VCID_2 = 666.09
// K2_CONSTANT_BAND_6_VCID_2 = 1282.71

__device__
double Radiancia ( double pixel){
    double rad = pixel * ganancia + offset ;

    if ( rad < 0)
        rad = 0;
    return ( rad );
}

__device__
double Temperature( double term){
    return ( K2 / ( log (( K1 / Radiancia(term))+1)));
}


__global__
void Thermic(unsigned char *imagen1,int filas, int columnas, unsigned char *resultado){

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;


    if((row < filas) && (col < columnas)){

        int pos = ((row*columnas)+col)*3;

        //conversion de kelvin a centigrados
        double T = Temperature((double)imagen1[pos])-273.15;
        
        if (T<=-5){//azul intenso -5°
            resultado[pos+BLUE] = 155;
            resultado[pos+GREEN] = 57;
            resultado[pos+RED] = 26;
        }
        else if (T>=-5 && T<0){//azul
                resultado[pos+BLUE] = 240;
                resultado[pos+GREEN] = 98;
                resultado[pos+RED] = 58; 
            }
            else if (T>=0 && T<5){//azulado
                resultado[pos+BLUE] = 255;
                resultado[pos+GREEN] = 255;
                resultado[pos+RED] = 3; 
            }
                else if (T>=5 && T<10){//verdoso
                    resultado[pos+BLUE] = 120;
                    resultado[pos+GREEN] = 145;
                    resultado[pos+RED] = 120;
                }

                    else if (T>=10 && T<15){//marfil
                        resultado[pos+BLUE] = 6;
                        resultado[pos+GREEN] = 252;
                        resultado[pos+RED] = 197;
                    }
                        else if (T>=15 && T<20){//amarillo claro
                            resultado[pos+BLUE] = 5;
                            resultado[pos+GREEN] = 203;
                            resultado[pos+RED] = 235;
                        }
                            else if (T>=20 && T<25){//amarillo
                                resultado[pos+BLUE] = 10;
                                resultado[pos+GREEN] = 234;
                                resultado[pos+RED] = 241;
                            }
                                else if (T>=25 && T<28){//anaranjado
                                    resultado[pos+BLUE] = 5;
                                    resultado[pos+GREEN] = 130;
                                    resultado[pos+RED] = 230;
                            }
                                    else if (T>=28){//rojo                                
                                        resultado[pos+BLUE] = 20;
                                        resultado[pos+GREEN] = 35;
                                        resultado[pos+RED] = 220;
                                    }
        }
    }


int main(int argc, char **argv){

    if(argc !=2){
        printf("Please, enter the paths of the Thermal images \n");
        return -1;
    }

    clock_t start, end;
    double time_used;
    start = clock();


    cudaError_t error = cudaSuccess;

    //input image host
    unsigned char *h_InputImage;
    //input images Device
    unsigned char *d_InputImage;

    //output image Host
    unsigned char *h_ImageResult;
    //output image Device
    unsigned char *d_ImageResult;

    char* imageName1 = argv[1];    

    Mat image1;

    image1 = imread(imageName1, CV_LOAD_IMAGE_COLOR);  
    
    //se obtienen los atributos de las imagenes

    Size s = image1.size();

    int width = s.width;
    int height = s.height;

    //Se carga la imagen
    h_InputImage = image1.data;

    ///////////reserve memory for Host and device //////////////////
    int size = sizeof(unsigned char)*width*height*image1.channels();


   //se reserva memoria para la imagen de salida en el host 
    h_ImageResult = (unsigned char*)malloc(size);


    //Imagen de entrada device
    error = cudaMalloc((void**)&d_InputImage,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_InputImage \n");
        exit(-1);
    }



    //Imagen de salida device

     error = cudaMalloc((void**)&d_ImageResult,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_ImageResult \n");
        exit(-1);
    }


    ////////////Copia de imagen de entrada del host al device/////////////////

    error = cudaMemcpy(d_InputImage, h_InputImage,size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_InputImage a d_InputImage \n");
        exit(-1);
    }

    //////SE lanza El kernel ////////////

    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);////bloque de 32 x 32 hilos = 1024 hilos
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    Thermic<<<dimGrid,dimBlock>>>(d_InputImage, height, width, d_ImageResult);

    cudaDeviceSynchronize();

    //copian los datos de la imagen del device a la de salida del host
    cudaMemcpy(h_ImageResult,d_ImageResult,size,cudaMemcpyDeviceToHost);


    Mat Thermal_Image;
    Thermal_Image.create(height,width,CV_8UC3);
    Thermal_Image.data = h_ImageResult;

    //imshow("Image Natural color",Thermal_Image);
	imwrite("./Thermal_Image.jpg",Thermal_Image);

    //waitKey(0);

    end = clock();
    time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf ("%lf \n",time_used);

    //Se libera memoria
    free(h_ImageResult);

    cudaFree(d_InputImage);

    cudaFree(d_ImageResult);
    
    return 0;
}
