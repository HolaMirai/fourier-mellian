#include "iostream"
#include "opencv2/opencv.hpp"

#include "CFourierMellin.h"
#include "CSegment.h"



using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

	if (argc != 2)
	{
		return 0;
	}
	Mat frame0, frame1, frame2, frame3, frame4, frame5;
	Mat rstImg0, rstImg1, rstImg2;
	Mat subImg0, subImg1;

	CFourierMellin fm;
	CSegment sg;

	VideoCapture cap(argv[1]);
	VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(2 * 640, 2 * 480));

	namedWindow("moveObject", 1);
	//namedWindow("video", 1);
	cap >> frame0;
	cap >> frame1;
	cap >> frame2;
	cap >> frame3;
	cap >> frame4;

	cvtColor(frame0, frame0, CV_BGR2GRAY);
	cvtColor(frame1, frame1, CV_BGR2GRAY);
	cvtColor(frame2, frame2, CV_BGR2GRAY);
	cvtColor(frame3, frame3, CV_BGR2GRAY);
	cvtColor(frame4, frame4, CV_BGR2GRAY);

	frame0.convertTo(frame0, CV_32FC1, 1.0 / 255.0);
	frame1.convertTo(frame1, CV_32FC1, 1.0 / 255.0);
	frame2.convertTo(frame2, CV_32FC1, 1.0 / 255.0);
	frame3.convertTo(frame3, CV_32FC1, 1.0 / 255.0);
	frame4.convertTo(frame4, CV_32FC1, 1.0 / 255.0);

	while (waitKey(5) != 'q')
	{
		static int i = 0;
		if (++i == 10) 
		{  
			printf(" \n"); }
		cap >> frame5;
		if (frame5.empty()) break;

		Mat frame5u = frame5.clone();
		cvtColor(frame5, frame5, CV_BGR2GRAY);
		
		frame5.convertTo(frame5, CV_32FC1, 1.0 / 255.0);
		medianBlur(frame5, frame5, 3);

		fm.imregTwice(frame0, frame3, rstImg0);
		fm.imregTwice(frame1, frame5, rstImg1);

		//GaussianBlur(frame1, frame1, Size(3, 3), 0.5);
		absdiff(frame0, rstImg0, subImg0);
		absdiff(frame1, rstImg1, subImg1);
		Mat result0 = subImg1.mul(subImg0);

		int b = 20;
		Mat Mask(result0.rows, result0.cols,  CV_32FC1, Scalar(0));
		Mask(Rect(b, b, result0.cols - 2 * b, result0.rows - 2 * b)) = Scalar(1);

		Mat result = Mask.mul(result0);

		Mat normResult;
		normalize(result, normResult, 255.0, 0.0, NORM_MINMAX);
		normResult.convertTo(normResult, CV_8UC1);

		// threshold
		Mat thresImg;
		threshold(normResult, thresImg, 20, 255, THRESH_BINARY);

		Mat dilateImgPre, erodeImg, dilateImg;

		Mat kernelDilatePre(7, 7, CV_8UC1, Scalar(1));
		Mat kernelErode(9, 9, CV_8UC1, Scalar(1));
		Mat kernelDilate(17, 17, CV_8UC1, Scalar(1));
		
		dilate(thresImg, dilateImgPre, kernelDilatePre);
		erode(dilateImgPre, erodeImg, kernelErode);
		dilate(erodeImg, dilateImg, kernelDilate);


		Mat vecReduceX, histImgX;
		reduce(dilateImg, vecReduceX, 0, CV_REDUCE_SUM, CV_MAT_DEPTH(CV_32F));
		sg.createHist(Size(640, 480), vecReduceX, histImgX, 0, true, 20, 20);

		cv::vector<cv::Point> lx;
		sg.searchLocate(vecReduceX, lx, 0);
		cv::vector<cv::Point>::iterator it, it_end;
		it = lx.begin();
		it_end = lx.end();
		for (;it != it_end; ++it)
		{
			CvScalar color = CV_RGB(rand() % 255, rand() % 255, rand() % 255);
			line(frame5u, Point(it->x, 0), Point(it->x, frame5u.rows), color);
			line(frame5u, Point(it->y, 0), Point(it->y, frame5u.rows), color);
		}

		// 确定了目标的X轴坐标，再来确定Y坐标
		Mat vecReduceY,histImgY;
		reduce(dilateImg, vecReduceY, 1, CV_REDUCE_SUM, CV_MAT_DEPTH(CV_32F));
		sg.createHist(Size(640, 480), vecReduceY, histImgY, 1, true, 20, 20);

		/*	cv::vector<cv::Point> ly;
			for (it = lx.begin; it != it_end; ++it)
			{
				reduce(dilateImg(Rect(it->x,0,it->y - it->x,dilateImg.rows)), vecReduceY, 1, CV_REDUCE_SUM, CV_MAT_DEPTH(CV_32F));
				sg.searchLocate(vecReduceY,ly,)

			}
	*/

		




		// 保存记录
		Mat saveImg(2 * 480, 640 * 2, CV_8UC1, Scalar(0));
		dilateImg.copyTo(saveImg(Rect(0, 0, 640, 480)));
		histImgX.copyTo(saveImg(Rect(0, 480, 640, 480)));
		histImgY.copyTo(saveImg(Rect(640, 480, 640, 480)));

		cvtColor(saveImg, saveImg, CV_GRAY2RGB);
		frame5u.copyTo(saveImg(Rect(640, 0, 640, 480)));
		imshow("moveObject", saveImg);
		writer << saveImg;


		frame0 = frame1;
		frame1 = frame2;
		frame2 = frame3;
		frame3 = frame5;
		frame4 = frame5;
	}
	writer.release();
	return 0;
}