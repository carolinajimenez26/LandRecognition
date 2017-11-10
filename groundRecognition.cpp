#include <tiffio.h>
#include <iostream>
#include <unordered_map>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#define dbg(x) cout << #x << ": " << x << endl
int d; // EARTH_SUN_DISTANCE
int theta_SE; // angle of sun elevation SUN_ELEVATION

using namespace std;

struct Band {
  // image m;
  int bandNumber;
  string fileName; // path of the image (FILE_NAME_BAND_N)
  unordered_map<int, vector<int>> m;
  int RADIANCE_MULT_BAND, RADIANCE_ADD_BAND, REFLECTANCE_MULT_BAND,
      REFLECTANCE_ADD_BAND, K1, K2;

  Band() {}
};

void read(string fileName, vector<Band> &v) { // Read MTL file
  ifstream infile(fileName);
  string line;
  while (getline(infile, line)) {
    if (line == "  GROUP = PRODUCT_METADATA") {
      string element = "";
      while (element != "  END_GROUP = PRODUCT_METADATA") {
        getline(infile, element);
        cout << "element " << element << endl;
      }
      cout << line << endl;
      string s;
      istringstream iss(line);
      iss >> s;
      cout << "s: " << s << endl;
    }
  }
}

/*
  L_lambda = (Ml)*(Qcal) + Al
  RADIANCE_MULT_BAND_N -> Ml
  RADIANCE_ADD_BAND_N -> Al
  Qcal -> current pixel of the band
*/
// int radiance(Band b, int pixel) {
//   return b.RADIANCE_MULT_BAND * b.m[pixel] + b.RADIANCE_ADD_BAND;
// }

/* with angular correction
  p_lambda = ((Mp)*(Qcal) + Ap) / sin(theta_SE)
  REFLECTANCE_MULT_BAND_N -> Mp
  REFLECTANCE_ADD_BAND_N -> Ap
  Qcal -> current pixel of the band
*/
// int reflectance(Band b, int pixel) {
//   return (b.REFLECTANCE_MULT_BAND * b.m[pixel] + b.REFLECTANCE_ADD_BAND) / sin(theta_SE);;
// }

/*
 T = K2 / (ln ( (K1 / L_lambda) + 1 ) )
*/
// int T(Band b, int pixel) {
//   return b.K2 / (log((b.K1 / radiance(b, pixel)) + 1));
// }

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: <images path> <mtl path>" << endl;
    return 1;
  }
  vector<Band> v;
  string imagesPath(argv[1]);
  string mtlPath(argv[2]);
  read(mtlPath, v);
  return 0;
}
