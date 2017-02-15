#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


int main00(int argc, char *argv[])
{
	
	Mat src = imread("lena.jpg", 1);

	medianBlur(src, src, 9);

	Mat dst;
	threshold(src, src, 50, 150, CV_THRESH_BINARY);
	Mat ker(9, 9, CV_8UC1, Scalar(1));
	erode(src, src, ker);
	

	

	return 0;
}
