#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

void convolv(Mat& src, Mat& kernerl) 
{
    Mat output_image;
    filter2D(src, output_image, -1, kernerl);
    std::cout << src << std::endl << kernerl << std::endl;
    std::cout << output_image << std::endl;

}
int main(){
    Mat input_image = (Mat_<uchar>(8, 8) <<
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 255, 255, 255, 0, 0, 0, 255,
        0, 255, 255, 255, 0, 0, 0, 0,
        0, 255, 255, 255, 0, 255, 0, 0,
        0, 0, 255, 0, 0, 0, 0, 0,
        0, 0, 255, 0, 0, 255, 255, 0,
        0, 255, 0, 255, 0, 0, 255, 0,
        0, 255, 255, 255, 0, 0, 0, 0);

    Mat input_image2 = (Mat_<uchar>(3, 3) <<
        1, 2, 3, 
        4, 5, 6,
        7, 8, 9);

    Mat kernel = (Mat_<int>(3, 3) <<
        0, 1, 0,
        1, -1, 1,
        0, 1, 0);
    Mat kernel2 = (Mat_<int>(3, 3) <<
        0, 1, 0,
        1, 0, 1,
        0, 1, 0);
    Mat kernel3 = (Mat_<int>(3, 3) <<
        0, 0, 0,
        0, -1, 0,
        0, 0, 0);
    Mat kernel5 = (Mat_<int>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);

    convolv(input_image2, kernel);

    Mat M = (Mat_<uchar>(3, 3) <<
        255, 255, 255,
        255, 255, 255,
        255, 255, 255);
    // convolv(M, kernel2);
    convolv(M, kernel5);

    return 0;
}
