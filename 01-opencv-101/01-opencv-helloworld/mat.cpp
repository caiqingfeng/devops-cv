#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv )
{
    Mat M(200, 200, CV_8UC3, Scalar(0, 255, 255));
    //cout << "M = " << endl << " " << M << endl << endl;

    randu(M, Scalar::all(0), Scalar::all(255));
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", M);
    waitKey(0);

    return 0;
}