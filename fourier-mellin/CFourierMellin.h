#pragma once

#include "cv.h"

class CFourierMellin
{
public:
	CFourierMellin();
	~CFourierMellin();

public:
	int imregOnce(cv::Mat& im0, cv::Mat& im1, cv::Mat &rstImg);
	int imregTwice(cv::Mat &im0, cv::Mat &im1, cv::Mat &rstImg);

private:
	
	
	void ForwardFFT(cv::Mat &Src, cv::Mat *FImg, bool do_recomb = true);
	void InverseFFT(cv::Mat *FImg, cv::Mat &Dst, bool do_recomb = true);

	void highpass(cv::Size sz, cv::Mat& dst);
	float logpolar(cv::Mat& src, cv::Mat& dst);

	void Recomb(cv::Mat &src, cv::Mat &dst);
};
