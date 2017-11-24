#include <stdio.h>
#include <tiffio.h>
#include <stdlib.h>
#include <iostream>
#define dbg(x) cout << #x << ": " << x << endl

using namespace std;

int main(int argc, char **argv) {
	TIFF *image;
	uint32 width, height;
	int row, col, imagesize;
	int nsamples;
	unsigned char *scanline=NULL;

	uint16 BitsPerSample; // establece el numero de bits que se utilizan para codificar cada uno de los pixeles
												// puede utilizar 8, 16, 32 o 64 bits por pixel
	uint16 SamplesPerPixel; // escala de grises -> 1, a color -> 3
	uint16 i;
	uint16 RowPerStrip; // número de filas de cada strip
	uint16 TileWidth; // numero de columnas en cada tile
	uint16 TileLength; // numero de filas en cada tile
	tsize_t scanline_size;
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

	//allocate memory for reading tif image
	// scanline = (unsigned char *)_TIFFmalloc(SamplesPerPixel * width);
	scanline_size = TIFFScanlineSize(image);
	scanline = (unsigned char *)_TIFFmalloc(scanline_size);
	if (scanline == NULL){
		fprintf (stderr,"Could not allocate memory!\n");
		exit(0);
	}

	FILE *f = fopen("tiffio_matrix.out", "w");
	for (row = 0; row < height; row++) {
		int n = TIFFReadScanline(image, scanline, row, 0); // gets all the row
		if(n==-1){
	    printf("Error");
	    return 1;
    }
		for (col = 0; col < width; col++) {
			// printf("%d ", scanline[col]);
			fprintf(f, "%d ", scanline[col]);
		}
		// printf("\n");
		fprintf(f, "\n");
	}

	fclose(f);

	_TIFFfree(scanline); //free allocate memory

	TIFFClose(image);

}
