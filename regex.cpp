// regex_search example
#include <iostream>
#include <string>
#include <regex>
#define dbg(x) cout << #x << ": " << x << endl

using namespace std;

int main ()
{
  string s = "FILE_NAME_BAND_1 = \"LT04_L1TP_009057_19890604_20170203_01_T1_B1.TIF\"\n";
  s += "FILE_NAME_BAND_2 = \"LT04_L1TP_009057_19890604_20170203_01_T1_B2.TIF\"\n";
  smatch m;
  regex e ("\\b(FILE_NAME_BAND_)([^ ]*)");   // matches words beginning by "FILE_NAME_BAND_"

  cout << "Target sequence: " << s << std::endl;
  cout << "Regular expression: /\\b(FILE_NAME_BAND_)([^ ]*)/" << endl;
  cout << "The following matches and submatches were found:" << endl;

  while (regex_search (s,m,e)) {
    for (auto x:m) {
      dbg(x);
    }
    s = m.suffix().str();
    dbg(s);
    cout << endl;
  }

  return 0;
}
