#include<iostream>
#include<stdio.h>
#include<malloc.h>
//#include <cv.h>
//#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define RED 2
#define GREEN 1
#define BLUE 0


#define FACTOR_MODIFY 2


void naturalColor(unsigned char *imagen1,unsigned char *imagen2,unsigned char *imagen3, int filas, int columnas, unsigned char *resultado){


	for(int i = 0; i < filas; i++){
		for(int j = 0; j < columnas; j++){//hacemos el recorrido por cada pixel

            int pos = ((i*columnas)+j)*3;
        
            resultado[pos+BLUE] = imagen1[pos+BLUE];
            resultado[pos+GREEN] = imagen2[pos+GREEN];
            resultado[pos+RED] = imagen3[pos+RED];
        }
	}
}


int main(int argc, char **argv){


    unsigned char *InputImageB1,*InputImageB2,*InputImageB3, *InputImageColor;
	unsigned char *ImageResult;


    if(argc !=4){
        printf("Please, enter the paths of the three images \n");
        return -1;
    }
	
	char* imageName1 = argv[1];
    char* imageName2 = argv[2];
    char* imageName3 = argv[3];
    

	Mat image1, image2, image3;

	// //times
    //  clock_t start, end;
    //  double time_used;

  	image1 = imread(imageName1, CV_LOAD_IMAGE_COLOR);
    image2 = imread(imageName2, CV_LOAD_IMAGE_COLOR);
    image3 = imread(imageName3, CV_LOAD_IMAGE_COLOR);    
    
    //se obtienen los atributos de las imagenes

    Size s = image1.size();


    //cout << "tamaño: "<< s;

    int width = s.width;
    int height = s.height;

    InputImageB1 = image1.data;
    InputImageB2 = image2.data;
    InputImageB3 = image3.data;

    //int size = sizeof(unsigned char)*width*height;//para la imagen en escala de grises
	int size = sizeof(unsigned char)*width*height*image1.channels();//imagen de salida natural color

    //se reserva memoria para la imagen de salida  
    ImageResult = (unsigned char*)malloc(size);

    naturalColor(InputImageB1,InputImageB2,InputImageB3, height, width, ImageResult);

    Mat ImageNaturalColor;
    ImageNaturalColor.create(height,width,CV_8UC3);
    ImageNaturalColor.data = ImageResult;

    //imshow("Image Natural color",ImageNaturalColor);
	imwrite("./naturalColor.jpg",ImageNaturalColor);

    waitKey(0);

    //Se libera memoria
    free(ImageResult);

    return 0;
}
