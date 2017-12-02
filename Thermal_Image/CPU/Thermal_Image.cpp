#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <time.h>

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

double Radiancia ( double pixel){
    double rad = pixel * ganancia + offset ;

    if ( rad < 0)
        rad = 0;
    return ( rad );
}

double Temperature( double term){
    return ( K2 / ( log (( K1 / Radiancia(term))+1)));
}


void Thermic(unsigned char *imagen1,int filas, int columnas, unsigned char *resultado){


	for(int i = 0; i < filas; i++){
		for(int j = 0; j < columnas; j++){//hacemos el recorrido por cada pixel

            int pos = ((i*columnas)+j)*3;

            double T = Temperature((double)imagen1[pos])-273.15;
            //cout <<T<<"-";

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
}


int main(int argc, char **argv){

    if(argc !=2){
        printf("Please, enter the paths of the three images \n");
        return -1;
    }

    unsigned char *InputImageB1;
    unsigned char *ImageResult;

    clock_t start, end;
    double time_used;
    start = clock();

	
    char* imageName1 = argv[1];    

	Mat image1;

  	image1 = imread(imageName1, 1);
   
    //se obtienen los atributos de las imagenes
    Size s = image1.size();

    int width = s.width;
    int height = s.height;

    InputImageB1 = image1.data;

	int size = sizeof(unsigned char)*width*height*image1.channels();//imagen de salida natural color

    //se reserva memoria para la imagen de salida  
    ImageResult = (unsigned char*)malloc(size);

    Thermic(InputImageB1, height, width, ImageResult);

    Mat ThermalImage;
    ThermalImage.create(height,width,CV_8UC3);
    ThermalImage.data = ImageResult;

    //imshow("Image Natural color",ImageNaturalColor);
	imwrite("./ThermalImage.jpg",ThermalImage);

    end = clock();
    time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf ("%lf \n",time_used);

    //Se libera memoria
    free(ImageResult);

    return 0;
}
