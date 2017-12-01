#include <tiffio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#define INF numeric_limits<int>::max()
#define dbg(x) cout << #x << ": " << x << endl
double d; // EARTH_SUN_DISTANCE
double theta_SE; // angle of sun elevation SUN_ELEVATION

using namespace std;

struct Band {
  // image m;
  int bandNumber;
  string fileName; // path of the image (FILE_NAME_BAND_N)
  // unordered_map<int, vector<int>> m;
  double RADIANCE_MULT_BAND, RADIANCE_ADD_BAND, REFLECTANCE_MULT_BAND,
      REFLECTANCE_ADD_BAND, K1, K2;

  Band() {
    fileName = "";
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

void read(string fileName, vector<Band> &v) { // Read MTL file
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
          v[i].setFileName(splitted[1]);
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
  double SUN_ELEVATION = 53.66309190, EARTH_SUN_DISTANCE = 1.0145756;
  vector<Band> v(7);
  string imagesPath(argv[1]);
  string mtlPath(argv[2]);
  read(mtlPath, v);
  cout << "------------" << endl;
  for (auto e : v) e.getInfo();
  return 0;
}
