#include<iostream>
#include<opencv2/opencv.hpp>
#define dbg(x) cout << #x << ": " << x << endl

using namespace std;
using namespace cv;

int main(int argc, char **argv){

  char* imageName = argv[1];

  if (argc !=2) {
    cout << "Path of the image must be specified!!" << endl;
    return 1;
  }

  Mat image = imread(imageName, 0);
  Mat copy = image.clone();

  if (!image.data) {
    printf("No image Data\n");
    return 1;
  }

  Size s = image.size();
  dbg(s);
  s = copy.size();
  dbg(s);

  dbg(image.channels());
  dbg(copy.channels());

  std::vector<uchar> buf;
  imencode(".TIF", copy, buf);

  imwrite( "../images/copy.TIF", copy);

  resize(image, image, Size(image.cols/10, image.rows/10));
  imshow("Image",image);

  resize(copy, copy, Size(copy.cols/10, copy.rows/10));
  imshow("Copy",copy);

  waitKey(0);

  return 0;
}
