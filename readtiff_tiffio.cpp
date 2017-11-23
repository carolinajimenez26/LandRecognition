#include <stdio.h>
#include <tiffio.h>
#include <stdlib.h>
#include <iostream>
#define dbg(x) cout << #x << ": " << x << endl

using namespace std;

int main(int argc, char **argv) {
	TIFF *image;
	uint32 width, height;
	int r1,c1, t1, imagesize;
	int nsamples;
	unsigned char *scanline=NULL;

	uint16 BitsPerSample; // establece el numero de bits que se utilizan para codificar cada uno de los pixeles
												// puede utilizar 8, 16, 32 o 64 bits por pixel
	uint16 SamplesPerPixel; // escala de grises -> 1, a color -> 3
	uint16 i;
	uint16 RowPerStrip; // número de filas de cada strip
	uint16 TileWidth; // numero de columnas en cada tile
	uint16 TileLength; // numero de filas en cada tile
	image = TIFFOpen(argv[1], "r");

	// Open the TIFF image
	if(image == NULL){
		cerr << "Could not open incoming image" << endl;
		return 1;
	}

	// Find the width and height of the image
	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	dbg(width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
	dbg(height);
	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);
	dbg(BitsPerSample);
	TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
	dbg(SamplesPerPixel);
	TIFFGetField(image, TIFFTAG_ROWSPERSTRIP, &RowPerStrip);
	dbg(RowPerStrip);
	TIFFGetField(image, TIFFTAG_TILEWIDTH, &TileWidth);
	dbg(TileWidth);
	TIFFGetField(image, TIFFTAG_TILELENGTH, &TileLength);
	dbg(TileLength);
	imagesize = height * width;	//get image size
	dbg(imagesize);

	// //allocate memory for reading tif image
	// scanline = (unsigned char *)_TIFFmalloc(SamplesPerPixel*width);
	// if (scanline == NULL){
	// 	fprintf (stderr,"Could not allocate memory!\n");
	// 	exit(0);
	// }
  //
	// for (r1 = 0; r1 < height; r1++) {
	// 	TIFFReadScanline(image, scanline, r1, 0);
	// 	for (c1 = 0; c1 < width; c1++) {
	// 		t1 = c1*SamplesPerPixel;
  //
	// 		for(i=0; i<SamplesPerPixel; i++)
	// 			printf("%u \t", *(scanline + t1+i));
	// 		printf("\n");
	// 	}
	// }
  //
	// _TIFFfree(scanline); //free allocate memory

	TIFFClose(image);

}
