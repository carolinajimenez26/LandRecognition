#include <tiffio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#include <tiffio.h>
#include <utility> // pair
#define INF numeric_limits<int>::max()
#define dbg(x) cout << #x << ": " << x << endl
double d; // EARTH_SUN_DISTANCE
double theta_SE; // angle of sun elevation SUN_ELEVATION in degrees

using namespace std;

struct Band {
  int bandNumber;
  string fileName; // path of the image (FILE_NAME_BAND_N)
  string radiance_fileName, reflectance_fileName, temperature_fileName,
  ndvi_fileName, ndwi_fileName;
  double RADIANCE_MULT_BAND, RADIANCE_ADD_BAND, REFLECTANCE_MULT_BAND,
      REFLECTANCE_ADD_BAND, K1, K2;

  Band() {
    fileName = "";
    radiance_fileName = "";
    reflectance_fileName = "";
    temperature_fileName = "";
    ndvi_fileName = "";
    ndwi_fileName = "";
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
      string element = "";
      while (element != "  END_GROUP = RADIOMETRIC_RESCALING") {
        for (int i = 0; i < 7; i++) { // RADIANCE_MULT_BAND
          dbg(i);
          getline(infile, element);
          dbg(element);
          vector<string> splitted = split(element, '=');
          dbg(splitted[1]);
          dbg(toDouble(splitted[1]));
          v[i].RADIANCE_MULT_BAND = toDouble(splitted[1]);
        }
        cout << "-------------" << endl;
        for (int i = 0; i < 7; i++) { // RADIANCE_ADD_BAND
          dbg(i);
          getline(infile, element);
          dbg(element);
          vector<string> splitted = split(element, '=');
          dbg(splitted[1]);
          dbg(toDouble(splitted[1]));
          v[i].RADIANCE_ADD_BAND = toDouble(splitted[1]);
        }
        cout << "-------------" << endl;
        for (int i = 0; i < 6; i++) { // REFLECTANCE_MULT_BAND
          dbg(i);
          getline(infile, element);
          dbg(element);
          vector<string> splitted = split(element, '=');
          dbg(splitted[1]);
          dbg(toDouble(splitted[1]));
          if (i == 5) i++; // put it in the next band
          v[i].REFLECTANCE_MULT_BAND = toDouble(splitted[1]);
        }
        cout << "-------------" << endl;
        for (int i = 0; i < 6; i++) { // REFLECTANCE_ADD_BAND
          dbg(i);
          getline(infile, element);
          dbg(element);
          vector<string> splitted = split(element, '=');
          dbg(splitted[1]);
          dbg(toDouble(splitted[1]));
          if (i == 5) i++; // put it in the next band
          v[i].REFLECTANCE_ADD_BAND = toDouble(splitted[1]);
        }
        cout << "-------------" << endl;
        cout << "FINISHED!" << endl;
        break;
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

TIFF* createImage(TIFF *image, string fileName) {
  uint32 height, width;
  uint16 SamplesPerPixel, BitsPerSample, PHOTOMETRIC;

  TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
  TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);
  TIFFGetField(image, TIFFTAG_PHOTOMETRIC, &PHOTOMETRIC);

  TIFF *tif = TIFFOpen(fileName.c_str(),"w");
  if (!tif) {
    fprintf (stderr,"Error opening tiff!\n");
    return NULL;
  }

  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, SamplesPerPixel);
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, BitsPerSample);
  TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC);
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, width * SamplesPerPixel));

  return tif;
}

pair<double,double> getMinAndMax(TIFF *image, int height, int width, double *buf) {
  double min = numeric_limits<double>::max();
  double max = numeric_limits<double>::min();
  // FIND THE MINIMUM AND MAXIMUM
  for (uint32 row = 0; row < height; row++) {
		int n = TIFFReadScanline(image, buf, row, 0); // gets all the row
		if (n == -1) {
			printf("Error");
			return make_pair(min, max);
		}
		for (int col = 0; col < width; col++) {
      if (buf[col] > max) max = buf[col];
      if (buf[col] < min) min = buf[col];
		}
	}
  return make_pair(min, max);
}

void normalize(string fileName, string normalized_fileName) {
  TIFF *image = TIFFOpen(fileName.c_str(),"r");
	if (!image) {
		fprintf (stderr,"Error opening tiff!\n");
		exit(0);
	}
  uint32 height, width;
  uint16 SamplesPerPixel, BitsPerSample, PHOTOMETRIC;

  TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);

  double *buf2 = NULL;

  buf2 = (double *)_TIFFmalloc(width * SamplesPerPixel * sizeof(double));
  if (buf2 == NULL){
    cerr << "Could not allocate memory!" << endl;
    return;
  }

  pair<double, double> p = getMinAndMax(image, height, width, buf2);

  TIFF *tif = createImage(image, normalized_fileName);

	if (!tif) return;

  double *buf = NULL;
  buf = (double *)_TIFFmalloc(width * SamplesPerPixel * sizeof(double));
  if (buf == NULL){
    cerr << "Could not allocate memory!" << endl;
    return;
  }

  for (uint32 row = 0; row < height; row++) {
		int n = TIFFReadScanline(image, buf2, row, 0); // gets all the row
		if (n == -1) {
			printf("Error");
      return;
		}
		for (int col = 0; col < width; col++) {
      double c = ((buf2[col] - p.first) * (255 / (p.second - p.first)));
      buf[col] = c;
    }
    TIFFWriteScanline(tif, buf, row, 0);
	}
  (void) TIFFClose(image);
  (void) TIFFClose(tif);
	if (buf) _TIFFfree(buf);
  if (buf2) _TIFFfree(buf2);
}

/*
  L_lambda = (Ml)*(Qcal) + Al
  RADIANCE_MULT_BAND_N -> Ml
  RADIANCE_ADD_BAND_N -> Al
  Qcal -> current pixel of the band
*/
double radiance(Band &b, int pixel) {
  return b.RADIANCE_MULT_BAND * pixel + b.RADIANCE_ADD_BAND;
}

void getRadiance(TIFF *image, Band &b, string path) {
  uint32 height, width;
  uint16 SamplesPerPixel;

  TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);

  string fileName = path + "radiance_B" + toString(b.bandNumber + 1) + ".TIF";
  TIFF *tif = createImage(image, fileName);

  if (tif == NULL) return;

  b.radiance_fileName = fileName;

  //----------------------------------------------------

	double *buf = NULL; // buffer used to store the row of pixel information for writing to file

  buf = (double *)_TIFFmalloc(width * SamplesPerPixel * sizeof(double));
	if (buf == NULL){
		cerr << "Could not allocate memory!" << endl;
		return;
	}

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
double reflectance(Band &b, int pixel) {
  double theta = (theta_SE * M_PI) / 180;
  double value = (b.REFLECTANCE_MULT_BAND * pixel + b.REFLECTANCE_ADD_BAND) / sin(theta);
  // return (value < 0 ? 0 : value);
  return value;
}

/*
  R_lambda = (pi * L_lambda * d^2) / ESUN_lambda * sin(theta_SE)
*/
double reflectance2(Band &b, int pixel, int rad_pixel) {
  double ESUN = 0.0, theta = (theta_SE * M_PI) / 180, value = 0;
  if (b.bandNumber == 0) ESUN = 1997;
  if (b.bandNumber == 1) ESUN = 1812;
  if (b.bandNumber == 2) ESUN = 1533;
  if (b.bandNumber == 3) ESUN = 1039;
  if (b.bandNumber == 4) ESUN = 230.8;
  if (b.bandNumber == 6) ESUN = 84.9;
  // value = (M_PI * radiance(b, pixel) * d * d) / (ESUN * sin(theta));
  value = (M_PI * rad_pixel * d * d) / (ESUN * sin(theta));
  // return (value < 0 ? 0 : value);
  return value;
}

void getReflectance(TIFF *image, Band &b, string path) {
  uint32 height, width;
  uint16 SamplesPerPixel;

	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);

  //------Creates image for radiance information---
  string fileName = path + "reflectance_B" + toString(b.bandNumber + 1) + ".TIF";
  TIFF *tif = createImage(image, fileName);
	if (!tif) return;

  b.reflectance_fileName = fileName;

	double *buf = NULL; // buffer used to store the row of pixel information for writing to file

  buf = (double *)_TIFFmalloc(width * SamplesPerPixel * sizeof(double));
	if (buf == NULL) {
		cerr << "Could not allocate memory!" << endl;
		return;
	}

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
double T(Band &b, int pixel) {
  return b.K2 / (log((b.K1 / radiance(b, pixel)) + 1));
}

void getTemperature(TIFF *image, Band &b, string path) {
  uint32 height, width;
  uint16 SamplesPerPixel;

	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);

  //------Creates image for radiance information---
  string fileName = path + "temperature_B" + toString(b.bandNumber + 1) + ".TIF";
  TIFF *tif = createImage(image, fileName);

  if (!tif) return;

  b.temperature_fileName = fileName;

	double *buf = NULL; // buffer used to store the row of pixel information for writing to file

  buf = (double *)_TIFFmalloc(width * SamplesPerPixel * sizeof(double));
	if (buf == NULL) {
		cerr << "Could not allocate memory!" << endl;
		return;
	}

	for (uint32 row = 0; row < height; row++) {
		int n = TIFFReadScanline(image, buf, row, 0); // gets all the row
		if (n == -1) {
			printf("Error");
			return;
		}
		for (int col = 0; col < width; col++) {
			buf[col] = T(b, buf[col]);
		}
    TIFFWriteScanline(tif, buf, row, 0);
	}
  (void) TIFFClose(tif);
	if (buf) _TIFFfree(buf);
}

double ndvi(Band &red, Band &infrared, int red_pixel, int infrared_pixel) {
  double red_value = reflectance(red, red_pixel),
         infrared_value = reflectance(infrared, infrared_pixel);
  return (infrared_value - red_value) / (infrared_value + red_value);
}

string getNDVI(TIFF *image_red, TIFF *image_infrared, Band &red, Band &infrared, string path) {
  uint32 height, width;
  uint16 SamplesPerPixel;

	TIFFGetField(image_red, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image_red, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image_red, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);

  string fileName = path + "nvdi.TIF";
  TIFF *tif = createImage(image_red, fileName);
	if (!tif) return "";

	unsigned char *buf_red = NULL, *buf_infrared = NULL; // buffer used to store the row of pixel information for writing to file
  double *buf = NULL;

  buf = (double *)_TIFFmalloc(width * SamplesPerPixel * sizeof(double));
	if (buf == NULL){
		cerr << "Could not allocate memory!" << endl;
		return "";
	}

  buf_red = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(image_red));
	if (buf_red == NULL){
		cerr << "Could not allocate memory!" << endl;
		return "";
	}

  buf_infrared = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(image_infrared));
	if (buf_infrared == NULL){
		cerr << "Could not allocate memory!" << endl;
		return "";
	}

	for (uint32 row = 0; row < height; row++) {
		int n = TIFFReadScanline(image_red, buf_red, row, 0); // gets all the row
    int m = TIFFReadScanline(image_infrared, buf_infrared, row, 0); // gets all the row
		if (n == -1 or m == -1) {
			printf("Error");
			return "";
		}
		for (int col = 0; col < width; col++) {
      buf[col] = ndvi(red, infrared, buf_red[col], buf_infrared[col]);
		}
    TIFFWriteScanline(tif, buf, row, 0);
	}
  (void) TIFFClose(tif);
	if (buf) _TIFFfree(buf);
  (void) TIFFClose(image_red);
	if (buf_red) _TIFFfree(buf_red);
	if (buf_infrared) _TIFFfree(buf_infrared);

  return fileName;
}

double ndwi(Band &infrared, Band &medium_infrared, int infrared_pixel,
            int medium_infrared_pixel) {
  double medium_infrared_value = reflectance(medium_infrared, medium_infrared_pixel),
         infrared_value = reflectance(infrared, infrared_pixel);
  return (infrared_value - medium_infrared_value) /
         (infrared_value + medium_infrared_value);
}

string getNDWI(TIFF *image_infrared, TIFF *image_medium_infrared, Band &infrared,
              Band &medium_infrared, string path) {
  uint32 height, width;
  uint16 SamplesPerPixel;

	TIFFGetField(image_infrared, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image_infrared, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image_infrared, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);

  //--------------------------------------------
  string fileName = path + "nwdi.TIF";
  TIFF *tif = createImage(image_infrared, fileName);
	if (!tif) return "";

	unsigned char *buf = NULL, *buf_medium_infrared = NULL, *buf_infrared = NULL; // buffer used to store the row of pixel information for writing to file
  double *buf2 = NULL;

  buf2 = (double *)_TIFFmalloc(width * SamplesPerPixel * sizeof(double));
	if (buf2 == NULL){
		cerr << "Could not allocate memory!" << endl;
		return "";
	}

  buf = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(tif));
	if (buf == NULL){
		cerr << "Could not allocate memory!" << endl;
		return "";
	}

  buf_medium_infrared = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(image_medium_infrared));
	if (buf_medium_infrared == NULL){
		cerr << "Could not allocate memory!" << endl;
		return "";
	}

  buf_infrared = (unsigned char *)_TIFFmalloc(TIFFScanlineSize(image_infrared));
	if (buf_infrared == NULL){
		cerr << "Could not allocate memory!" << endl;
		return "";
	}
  
	for (uint32 row = 0; row < height; row++) {
		int n = TIFFReadScanline(image_medium_infrared, buf_medium_infrared, row, 0); // gets all the row
    int m = TIFFReadScanline(image_infrared, buf_infrared, row, 0); // gets all the row
		if (n == -1 or m == -1) {
			printf("Error");
			return "";
		}
		for (int col = 0; col < width; col++) {
      buf2[col] = ndwi(infrared, medium_infrared, buf_infrared[col], buf_medium_infrared[col]);
		}
    TIFFWriteScanline(tif, buf2, row, 0);
	}
  (void) TIFFClose(tif);
	if (buf) _TIFFfree(buf);
  if (buf2) _TIFFfree(buf2);
  (void) TIFFClose(image_medium_infrared);
	if (buf_medium_infrared) _TIFFfree(buf_medium_infrared);
	if (buf_infrared) _TIFFfree(buf_infrared);

  return fileName;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: <images path> <mtl path>" << endl;
    return 1;
  }
  vector<Band> v(7);
  string imagesPath(argv[1]);
  string mtl_fileName(argv[2]);
  string ndvi_path = "", ndwi_path = "";
  read(mtl_fileName, v, imagesPath);
  cout << "------------" << endl;
  for (auto e : v) e.getInfo();

  TIFF *image, *image_tmp;
  Band b_tmp;
  bool err = false;

  for (int i = 0; i < v.size(); i++) { // se calcula lo mismo para todas las bandas
    dbg(v[i].bandNumber);
    image = TIFFOpen(v[i].fileName.c_str(), "r");
    if(image == NULL){
  		cerr << "Could not open incoming image of band " << v[i].bandNumber << endl;
  		continue;
  	}
    getRadiance(image, v[i], imagesPath);
    normalize(v[i].radiance_fileName, imagesPath + "radiance_normalized_" +
              toString(v[i].bandNumber) + ".TIF");
    if (v[i].bandNumber == 5) { // la banda 6 se le aplica otro tipo de procesamiento
      getTemperature(image, v[i], imagesPath); // temperatura física del terreno
      normalize(v[i].temperature_fileName, imagesPath + "temperature_normalized_" +
                toString(v[i].bandNumber) + ".TIF");
    } else {
      getReflectance(image, v[i], imagesPath);
      normalize(v[i].reflectance_fileName, imagesPath + "reflectance_normalized_" +
                toString(v[i].bandNumber) + ".TIF");
    }
    if (v[i].bandNumber == 2) {
      image_tmp = image;
      if(image_tmp == NULL){
    		cerr << "Error copying image to image_red" << endl;
        err = true;
    	}
      b_tmp = v[i];
    }
    if (v[i].bandNumber == 3 and !err) {
      ndvi_path = getNDVI(image_tmp, image, b_tmp, v[i], imagesPath);
      dbg(ndvi_path);
      image_tmp = image;
      if(image_tmp == NULL){
    		cerr << "Error copying image to image_red" << endl;
        err = true;
    	}
      b_tmp = v[i];
    }
    if (v[i].bandNumber == 4 and !err) {
      ndwi_path = getNDWI(image_tmp, image, b_tmp, v[i], imagesPath);
      dbg(ndwi_path);
    }
  }
  normalize(ndvi_path, imagesPath + "ndvi_normalized.TIF");
  normalize(ndwi_path, imagesPath + "ndwi_normalized.TIF");
  TIFFClose(image);

  return 0;
}
