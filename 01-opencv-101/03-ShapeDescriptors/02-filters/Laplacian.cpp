/**
 * @file Morphology_1.cpp
 * @brief Erosion and Dilation sample code
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/// Global variables
Mat src, dst, src_gray;

const char* window_name = "filter2D Demo";
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

/**
 * @function main
 */
int main( int argc, char** argv )
{
  /// Load an image
  CommandLineParser parser( argc, argv, "{@input | ../data/lena.jpg | input image}" );
  src = imread( parser.get<String>( "@input" ), IMREAD_COLOR );
  if( src.empty() )
  {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to grayscale
  Mat abs_dst;
  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, abs_dst );
  imshow( window_name, abs_dst );
  waitKey(0);
  return 0;

}

