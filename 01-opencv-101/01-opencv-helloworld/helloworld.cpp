#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char** argv )
{
    const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";

    Mat image;
    image = imread( filename, IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    std::cout << "depth():" << image.depth() << std::endl;
    std::cout << "elemSize():" << image.elemSize() << std::endl;
    std::cout << "elemSize1():" << image.elemSize1() << std::endl;
    std::cout << "type():" << image.type() << std::endl;
    std::cout << "dims():" << image.dims << std::endl;
    std::cout << "rows():" << image.rows << std::endl;
    std::cout << "cols():" << image.cols << std::endl;
    std::cout << "channels():" << image.channels() << std::endl;
    std::cout << "step[0]:" << image.step[0] << std::endl;
    std::cout << "step[1]:" << image.step[1] << std::endl;

    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    // imshow("Display Image", image);
    // waitKey(0);
    return 0;
}
