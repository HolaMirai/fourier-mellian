#pragma once
#include "cv.h"

class CSegment
{
public:
	CSegment();
	~CSegment();

public:
	int createHist(cv::Size size, cv::Mat &binValue, cv::Mat &histImg, bool xy = 0, bool smooth = true, int cutSize = 20, int smoothSize = 10);

	int searchLocate(cv::Mat &vecReduce, cv::vector <cv::Point> &vecLocate, bool xy);

	int saveVideo(cv::Mat &srcImg, cv::Mat &binImg, cv::Mat &histImgX, cv::Mat &histImgY);
};