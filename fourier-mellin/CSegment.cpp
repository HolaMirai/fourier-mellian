#include "CSegment.h"
#include "opencv2/opencv.hpp"
using namespace cv;


CSegment::CSegment()
{
}

CSegment::~CSegment()
{
}

// xy = 0 向x轴投影
// xy = 1 向y轴投影
int CSegment::createHist(cv::Size size, cv::Mat &binValue, cv::Mat &histImg, bool xy, bool smooth, int cutSize, int smoothSize)
{
	int w = size.width;
	int h = size.height;

	Mat img(size, CV_32FC1);

	if (!xy)  // 向x轴投影
	{
		assert(binValue.cols == w && binValue.rows == 1);
		normalize(binValue, binValue, h, 0, NORM_MINMAX);
		for (int i = 0; i < w; ++i)
		{
			int val = (int)binValue.at<float>(0, i);
			img(Rect(i, 0, 1, val)) = Scalar(255);
		}
		flip(img, img, 0);
		Mat kerMatX(1, smoothSize, CV_8UC1, Scalar(1));
		dilate(img, img, kerMatX);
	}
	else
	{
		assert(binValue.rows == h && binValue.cols == 1);
		normalize(binValue, binValue, w, 0, NORM_MINMAX);

		for (int i = 0; i < h; ++i)
		{
			int val = (int)binValue.at<float>(i, 0);
			img(Rect(0, i, val, 1)) = Scalar(255);
		}

		Mat kerMatY(smoothSize, 1, CV_8UC1, Scalar(1));
		dilate(img, img, kerMatY);
	}

	Mat Mask(size, CV_32FC1, Scalar(0));
	Mask(Rect(cutSize, cutSize, size.width - 2 * cutSize, size.height - 2 * cutSize)) = Scalar(1);

	histImg = Mask.mul(img);
	histImg.convertTo(histImg, CV_8UC1);

	return 0;
}

// xy = 0 行向量
// xy = 1 列向量
// point.x 存储目标开始位置
// point.y 存储目标结束位置

int CSegment::searchLocate(cv::Mat & vecReduce, cv::vector<cv::Point> &vecLocate, bool xy)
{
	assert(vecReduce.type() == CV_32FC1);
	if (!xy)
	{
		int w = vecReduce.cols;
		float pre = 0;
		cv::Point p(0, 0);
		for (int i = 0; i < w; ++i)
		{
			float val = vecReduce.at<float>(0, i);
			if (pre == 0 && val > 0)
			{
				p.x = i;   //(y, x)
			}
			else if (pre > 0 && val == 0)
			{
				p.y = i;
				vecLocate.push_back(p);
			}
			pre = val;
		}
	}

	return 0;
}


int CSegment::saveVideo(cv::Mat & srcImg, cv::Mat & binImg, cv::Mat & histImgX, cv::Mat & histImgY)
{
	return 0;
}
