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

int clamp(int const &value){
	if(value >= 250)
		return 250;
	else
		return value;
}

void colorImage(TIFF *blue, TIFF *green, TIFF *red, uint32 height, uint32 width){
	tsize_t scanline_size_blue = TIFFScanlineSize(blue);
	tsize_t scanline_size_green = TIFFScanlineSize(green);
	tsize_t scanline_size_red = TIFFScanlineSize(red);

	unsigned char *scanline_blue = NULL, *buf_blue = NULL;
	unsigned char *scanline_green = NULL, *buf_green = NULL;
	unsigned char *scanline_red = NULL, *buf_red = NULL;

	scanline_blue = (unsigned char *)_TIFFmalloc(scanline_size_blue);
	scanline_green = (unsigned char *)_TIFFmalloc(scanline_size_green);
	scanline_red = (unsigned char *)_TIFFmalloc(scanline_size_red);

	buf_blue = (unsigned char *)_TIFFmalloc(scanline_size_blue);
	buf_green = (unsigned char *)_TIFFmalloc(scanline_size_green);
	buf_red = (unsigned char *)_TIFFmalloc(scanline_size_red);


	if (scanline_blue == NULL or scanline_green == NULL or scanline_red == NULL){
		fprintf (stderr,"Could not allocate memory!\n");
		exit(0);
	}

	//------------- for outImage -----------------------------------------------
	
	uint16 SamplesPerPixel, BitsPerSample;

	TIFFGetField(blue, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
	TIFFGetField(blue, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);

	TIFF *tif = TIFFOpen("images/outImage.TIFF","w");

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

	tsize_t linebytes = SamplesPerPixel * width; // length in memory of one row of pixel in the image.
	unsigned char *buf = NULL; // buffer used to store the row of pixel information for writing to file

	buf = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(tif));

	if (buf == NULL){
		fprintf (stderr,"Could not allocate memory!\n");
		exit(0);
	}

	// We set the strip size of the file to be size of one row of pixels
	TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, width * SamplesPerPixel));
	//--------------------------------------------------------------------------


	// FILE *f = fopen("tiffio_matrix.out", "w");
	for (int row = 0; row < height; row++) {
		int m = TIFFReadScanline(blue, scanline_blue, row, 0); // gets all the row
		int n = TIFFReadScanline(green, scanline_green, row, 0); // gets all the row
		int p = TIFFReadScanline(red, scanline_red, row, 0); // gets all the row

		if (n == -1 or m == -1 or p == -1) {
			printf("Error");
			exit(1);
		}

		for (int col = 0; col < width; col++) {
			//printf("%d ", scanline[col]);
			// fprintf(f, "%d ", scanline[col]);
			int pixel = scanline_blue[col] + scanline_green[col] + scanline_red[col]; 
			buf[col] = clamp(pixel);
		}
		TIFFWriteScanline(tif, buf, row, 0);
		printf("\n");
		// fprintf(f, "\n");
	}

	// fclose(f);

	_TIFFfree(scanline_blue); //free allocate memory
	_TIFFfree(scanline_green); //free allocate memory
	_TIFFfree(scanline_red); //free allocate memory
	(void) TIFFClose(tif);
	if (buf) _TIFFfree(buf);
	if (buf_blue) _TIFFfree(buf_blue);
	if (buf_green) _TIFFfree(buf_green);
	if (buf_red) _TIFFfree(buf_red);
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
	TIFF *blue;
	TIFF *green;
	TIFF *red;
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
	blue = TIFFOpen(argv[1], "r");
	green = TIFFOpen(argv[2], "r");
	red = TIFFOpen(argv[3], "r");

	// Open the TIFF image
	if(blue == NULL or green == NULL or red == NULL){
		cerr << "Could not open incoming image" << endl;
		return 1;
	}

	// Find the width and height of the image
	TIFFGetField(blue, TIFFTAG_IMAGEWIDTH, &width);
	dbg(width);
	TIFFGetField(blue, TIFFTAG_IMAGELENGTH, &height);
	dbg(height);
	TIFFGetField(blue, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);
	dbg(BitsPerSample);
	TIFFGetField(blue, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
	dbg(SamplesPerPixel);
	TIFFGetField(blue, TIFFTAG_ROWSPERSTRIP, &RowPerStrip);
	dbg(RowPerStrip);
	TIFFGetField(blue, TIFFTAG_TILEWIDTH, &TileWidth);
	dbg(TileWidth);
	TIFFGetField(blue, TIFFTAG_TILELENGTH, &TileLength);
	dbg(TileLength);
	imagesize = height * width;	//get image size
	dbg(imagesize);

	// print(image, height, width);
	//write(image, "./images/out.tiff");
	colorImage(blue, green, red, height, width);
	TIFFClose(blue);
	TIFFClose(green);
	TIFFClose(red);
}
