#include <stdio.h>
#include <tiffio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#define dbg(x) cout << #x << ": " << x << endl

using namespace std;

void print(TIFF *image, uint32 height, uint32 width) {
	//allocate memory for reading tif image
	// scanline = (unsigned char *)_TIFFmalloc(SamplesPerPixel * width);
	tsize_t scanline_size = TIFFScanlineSize(image);
	unsigned char *scanline = NULL, *buf = NULL;

	scanline = (unsigned char *)_TIFFmalloc(scanline_size);
	buf = (unsigned char *)_TIFFmalloc(scanline_size);
	if (scanline == NULL){
		fprintf (stderr,"Could not allocate memory!\n");
		exit(0);
	}

	// FILE *f = fopen("tiffio_matrix.out", "w");
	for (int row = 0; row < height; row++) {
		int n = TIFFReadScanline(image, scanline, row, 0); // gets all the row
		if (n == -1) {
			printf("Error");
			exit(1);
		}
		for (int col = 0; col < width; col++) {
			printf("%d ", scanline[col]);
			// fprintf(f, "%d ", scanline[col]);
		}
		printf("\n");
		// fprintf(f, "\n");
	}

	// fclose(f);

	_TIFFfree(scanline); //free allocate memory
}

void write(TIFF *image, const char *fileName) {
	uint32 height, width;
	uint16 SamplesPerPixel, BitsPerSample;

	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);

	TIFF *tif = TIFFOpen(fileName,"w");
	if (!tif) {
		fprintf (stderr,"Error opening tiff!\n");
		exit(0);
	}
	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, SamplesPerPixel);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, BitsPerSample);
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
	// TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 0);

	tsize_t linebytes = SamplesPerPixel * width; // length in memory of one row of pixel in the image.
	unsigned char *buf = NULL; // buffer used to store the row of pixel information for writing to file

	// Allocating memory to store the pixels of current row
	// if (TIFFScanlineSize(tif)linebytes)
	// 	buf = (unsigned char *)_TIFFmalloc(linebytes);
	// else
	buf = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(tif));

	if (buf == NULL){
		fprintf (stderr,"Could not allocate memory!\n");
		exit(0);
	}

	// We set the strip size of the file to be size of one row of pixels
	TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, width * SamplesPerPixel));

	//Now writing image to the file one strip at a time
	for (uint32 row = 0; row < height; row++) {
		// memcpy(buf, &image[(height - row - 1) * linebytes], linebytes); // check the index here, and figure out why not using h*linebytes
		int n = TIFFReadScanline(image, buf, row, 0);
		if (n == -1) {
			printf("Error");
			exit(1);
		}
		// if (TIFFWriteScanline(tif, buf, row, 0) < 0) break;
		TIFFWriteScanline(tif, buf, row, 0);
	}
	(void) TIFFClose(tif);
	if (buf) _TIFFfree(buf);
}

int main(int argc, char **argv) {
	TIFF *image;
	uint32 width, height;
	int row, col, imagesize;
	int nsamples;

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

	// print(image, height, width);
	// write(image, "./images/out.tiff");
	TIFFClose(image);
}
