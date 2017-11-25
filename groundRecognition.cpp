#include <tiffio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#include <tiffio.h>
#define INF numeric_limits<int>::max()
#define dbg(x) cout << #x << ": " << x << endl
double d; // EARTH_SUN_DISTANCE
double theta_SE; // angle of sun elevation SUN_ELEVATION

using namespace std;

struct Band {
  int bandNumber;
  string fileName; // path of the image (FILE_NAME_BAND_N)
  string radiance_fileName, reflectance_fileName;
  double RADIANCE_MULT_BAND, RADIANCE_ADD_BAND, REFLECTANCE_MULT_BAND,
      REFLECTANCE_ADD_BAND, K1, K2;

  Band() {
    fileName = "";
    radiance_fileName = "";
    reflectance_fileName = "";
    RADIANCE_MULT_BAND = -INF;
    RADIANCE_ADD_BAND = -INF;
    REFLECTANCE_MULT_BAND = -INF;
    REFLECTANCE_ADD_BAND = -INF;
    K1 = -INF;
    K2 = -INF;
  }

  void setFileName(string s) {
    fileName = s;
  }

  void getInfo() {
    cout << "------------------" << endl;
    cout << "Band " << bandNumber << endl;
    dbg(fileName);
    dbg(RADIANCE_MULT_BAND);
    dbg(RADIANCE_ADD_BAND);
    dbg(REFLECTANCE_MULT_BAND);
    dbg(REFLECTANCE_ADD_BAND);
    dbg(K1);
    dbg(K2);
    cout << "------------------" << endl;
  }
};

vector<string> split(string s, char tok) { // split a string by a token especified
  istringstream ss(s);
  string token;
  vector<string> v;

  while(getline(ss, token, tok)) {
    v.push_back(token);
  }

  return v;
}

double toDouble(string s) {
  stringstream ss;
  ss << s;
  double out;
  ss >> out;
  return out;
}

string toString(int n) {
  stringstream ss;
  ss << n;
  string out;
  ss >> out;
  return out;
}

void read(string fileName, vector<Band> &v, string path) { // Read MTL file
  ifstream infile(fileName);
  string line;
  while (getline(infile, line)) {
    if (line == "  GROUP = PRODUCT_METADATA") {
      smatch m;
      regex e("\\b(FILE_NAME_BAND_)([^ ]*)");
      string element = "", s = "";
      int i = 0;
      while (element != "  END_GROUP = PRODUCT_METADATA" and i < v.size()) {
        getline(infile, element);
        s = element;
        while (regex_search (s,m,e)) {
          s = m.suffix().str();
          vector<string> splitted = split(element, '"');
          v[i].setFileName(path + splitted[1]);
          v[i].bandNumber = i;
          i++;
        }
      }
    } else if (line == "  GROUP = IMAGE_ATTRIBUTES") {
      smatch m;
      regex e("\\b(SUN_ELEVATION = )([^ ]*)");
      regex e2("\\b(EARTH_SUN_DISTANCE = )([^ ]*)");
      string element = "", s = "";
      bool end = false, aux = false;
      while (element != "  END_GROUP = IMAGE_ATTRIBUTES" and not end) {
        getline(infile, element);
        s = element;
        if (!aux) {
          while (regex_search (s,m,e)) {
            s = m.suffix().str();
            vector<string> splitted = split(element, '=');
            theta_SE = toDouble(splitted[1]);
            dbg(theta_SE);
            aux = true;
          }
        } else {
          while (regex_search (s,m,e2)) {
            s = m.suffix().str();
            vector<string> splitted = split(element, '=');
            d = toDouble(splitted[1]);
            dbg(d);
            end = true;
          }
        }
      }
    } else if (line == "  GROUP = RADIOMETRIC_RESCALING") { // ARREGLAR
      smatch m;
      int i = 0, l = 0;
      vector<string> exps = {
        "RADIANCE_MULT_BAND",
        "RADIANCE_ADD_BAND",
        "REFLECTANCE_MULT_BAND",
        "REFLECTANCE_ADD_BAND"
      };
      string curr = exps[l];
      string total_exp = "\\b(" + curr + "_)([^ ]*)";
      regex e(total_exp);
      string element = "", s = "";
      while (element != "  END_GROUP = RADIOMETRIC_RESCALING") {
        if (i >= v.size()) {
          i = 0;
          l++;
          string curr = exps[l];
          string total_exp = "\\b(" + curr + "_)([^ ]*)";
          e = total_exp;
        }
        getline(infile, element);
        s = element;
        while (regex_search (s,m,e)) {
          s = m.suffix().str();
          vector<string> splitted = split(element, '=');
          if (l == 0) v[i++].RADIANCE_MULT_BAND = toDouble(splitted[1]);
          if (l == 1) v[i++].RADIANCE_ADD_BAND = toDouble(splitted[1]);
          if (l == 2) {
            if (i == 5) {
              v[++i].REFLECTANCE_MULT_BAND = toDouble(splitted[1]);
              i += 2;
            } else v[i++].REFLECTANCE_MULT_BAND = toDouble(splitted[1]);
          }
          if (l == 3) {
            if (i == 5) {
              v[++i].REFLECTANCE_ADD_BAND = toDouble(splitted[1]);
              i += 2;
            } else v[i++].REFLECTANCE_ADD_BAND = toDouble(splitted[1]);
          }
        }
      }
    } else if (line == "  GROUP = THERMAL_CONSTANTS") {
      string element;
      getline(infile, element);
      vector<string> splitted = split(element, '=');
      v[5].K1 = toDouble(splitted[1]);
      getline(infile, element);
      splitted = split(element, '=');
      v[5].K2 = toDouble(splitted[1]);
    }
  }
}

/*
  L_lambda = (Ml)*(Qcal) + Al
  RADIANCE_MULT_BAND_N -> Ml
  RADIANCE_ADD_BAND_N -> Al
  Qcal -> current pixel of the band
*/
int radiance(Band &b, int pixel) {
  return b.RADIANCE_MULT_BAND * pixel + b.RADIANCE_ADD_BAND;
}

void getRadiance(TIFF *image, Band &b, string path) {
  uint32 height, width;
  uint16 SamplesPerPixel, BitsPerSample;

	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
  TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);

  //------Creates image for radiance information---
  string fileName = path + "radiance_B" + toString(b.bandNumber + 1);
  TIFF *tif = TIFFOpen(fileName.c_str(),"w");
	if (!tif) {
		fprintf (stderr,"Error opening tiff!\n");
		exit(0);
	}
  b.radiance_fileName = fileName;

  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, SamplesPerPixel);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, BitsPerSample);
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);

  //----------------------------------------------------

  tsize_t linebytes = SamplesPerPixel * width; // length in memory of one row of pixel in the image.
	unsigned char *buf = NULL; // buffer used to store the row of pixel information for writing to file

  buf = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(tif));
	if (buf == NULL){
		cerr << "Could not allocate memory!" << endl;
		return;
	}

  // We set the strip size of the file to be size of one row of pixels
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, width * SamplesPerPixel));

	for (uint32 row = 0; row < height; row++) {
		int n = TIFFReadScanline(image, buf, row, 0); // gets all the row
		if (n == -1) {
			printf("Error");
			return;
		}
		for (int col = 0; col < width; col++) {
			buf[col] = radiance(b, buf[col]);
		}
    TIFFWriteScanline(tif, buf, row, 0);
	}
  (void) TIFFClose(tif);
	if (buf) _TIFFfree(buf);
}

/* with angular correction
  p_lambda = ((Mp)*(Qcal) + Ap) / sin(theta_SE)
  REFLECTANCE_MULT_BAND_N -> Mp
  REFLECTANCE_ADD_BAND_N -> Ap
  Qcal -> current pixel of the band
*/
int reflectance(Band &b, int pixel) {
  return (b.REFLECTANCE_MULT_BAND * pixel + b.REFLECTANCE_ADD_BAND) / sin(theta_SE);;
}

void getReflectance(TIFF *image, Band &b, string path) {
  uint32 height, width;
  uint16 SamplesPerPixel, BitsPerSample;

	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
  TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);

  //------Creates image for radiance information---
  string fileName = path + "reflectance_B" + toString(b.bandNumber + 1);
  TIFF *tif = TIFFOpen(fileName.c_str(),"w");
	if (!tif) {
		fprintf (stderr,"Error opening tiff!\n");
		exit(0);
	}
  b.radiance_fileName = fileName;

  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, SamplesPerPixel);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, BitsPerSample);
	TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);

  //----------------------------------------------------

  tsize_t linebytes = SamplesPerPixel * width; // length in memory of one row of pixel in the image.
	unsigned char *buf = NULL; // buffer used to store the row of pixel information for writing to file

  buf = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(tif));
	if (buf == NULL){
		cerr << "Could not allocate memory!" << endl;
		return;
	}

  // We set the strip size of the file to be size of one row of pixels
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, width * SamplesPerPixel));

	for (uint32 row = 0; row < height; row++) {
		int n = TIFFReadScanline(image, buf, row, 0); // gets all the row
		if (n == -1) {
			printf("Error");
			return;
		}
		for (int col = 0; col < width; col++) {
			buf[col] = reflectance(b, buf[col]);
		}
    TIFFWriteScanline(tif, buf, row, 0);
	}
  (void) TIFFClose(tif);
	if (buf) _TIFFfree(buf);
}

/*
 T = K2 / (ln ( (K1 / L_lambda) + 1 ) )
*/
// int T(Band b, int pixel) {
//   return b.K2 / (log((b.K1 / radiance(b, pixel)) + 1));
// }

void getTemperature(TIFF *image, Band b, string path) {

}

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: <images path> <mtl path>" << endl;
    return 1;
  }
  vector<Band> v(7);
  string imagesPath(argv[1]);
  string mtl_fileName(argv[2]);
  read(mtl_fileName, v, imagesPath);
  // cout << "------------" << endl;
  // for (auto e : v) e.getInfo();

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
  TIFF *image;

  for (int i = 0; i < 1; i++) { // se calcula lo mismo para todas las bandas
    image = TIFFOpen(v[i].fileName.c_str(), "r");
    if(image == NULL){
  		cerr << "Could not open incoming image of band " << v[i].bandNumber << endl;
  		continue;
  	}
    getRadiance(image, v[i], "./images/");
    if (v[i].bandNumber == 5) { // la banda 6 se le aplica otro tipo de procesamiento
      getTemperature(image, v[i], "./images/"); // temperatura física del terreno
    } else {
      getReflectance(image, v[i], "./images/");
    }
    TIFFClose(image);
  }

  return 0;
}
