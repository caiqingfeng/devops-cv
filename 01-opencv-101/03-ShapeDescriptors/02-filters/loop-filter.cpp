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
Mat src, dst;

int const max_elem = 2;
int const max_kernel_size = 21;
const char* window_name = "filter2D Demo";

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

  Mat kernel;
  Point anchor;
  double delta;
  int ddepth;
  int kernel_size;

  anchor = Point( -1, -1 );
  delta = 0;
  ddepth = -1;
  int ind = 0;
  for(;;)
       {
         char c = (char)waitKey(500);
         if( c == 27 )
           { break; }
         kernel_size = 3 + 2*( ind%5 );
         kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
         filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
         imshow( window_name, dst );
         ind++;
       }
  return 0;

}

