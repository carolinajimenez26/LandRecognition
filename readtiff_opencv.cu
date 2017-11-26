#include<iostream>
#include<opencv2/opencv.hpp>
#define dbg(x) cout << #x << ": " << x << endl

using namespace std;
using namespace cv;

void loop(Mat &image) {
  FILE *f = fopen("opencv_matrix.out", "w");
  for(int i = 0; i < image.rows; i++) {
    const unsigned char* Mi = image.ptr<unsigned char>(i);
    for(int j = 0; j < image.cols; j++) fprintf(f, "%d ", Mi[j]); //cout << (int)Mi[j] << " ";
    fprintf(f, "\n"); // cout << endl;
  }
  fclose(f);
}

__global__
void loopCU(cuda::GpuMat image) {
  printf("%sLOOPCU\n");
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  // printf("%d",image.rows);
  // if((row < image.rows) && (col < image.cols)) {
  //   const unsigned char* Mi = image.ptr<unsigned char>(row);
  //   cout << (int)image[row * image.cols + col] << " ";
  // }
}

int main(int argc, char **argv){

  char* imageName = argv[1];

  if (argc !=2) {
    cout << "Path of the image must be specified!!" << endl;
    return 1;
  }

  Mat image = imread(imageName, 0);
  // Mat copy = image.clone();
  // cuda::GpuMat other(image);

  if (!image.data) {
    printf("No image Data\n");
    return 1;
  }

  Size s = image.size();
  dbg(s);
  // s = copy.size();
  // dbg(s);

  dbg(image.channels());
  // dbg(copy.channels());

  // std::vector<uchar> buf;
  // imencode(".TIF", copy, buf);

  // imwrite( "../images/copy.TIF", copy);
  //
  // resize(image, image, Size(image.cols/10, image.rows/10));
  // imshow("Image",image);
  //
  // resize(copy, copy, Size(copy.cols/10, copy.rows/10));
  // imshow("Copy",copy);
  //
  // waitKey(0);

  loop(image);

  // int blockSize = 32;
  // dim3 dimBlock(blockSize, blockSize, 1);
  // dim3 dimGrid(ceil(image.cols/float(blockSize)), ceil(image.rows/float(blockSize)), 1);
  // loopCU<<<dimGrid,dimBlock>>>(other);
  // cudaDeviceSynchronize();

  return 0;
}
