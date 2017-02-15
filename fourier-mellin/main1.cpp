//#include "opencv2/opencv.hpp"
//#include "iostream"
//using namespace std;
//using namespace cv;
//
////----------------------------------------------------------
//// Recombinate image quaters
////----------------------------------------------------------
//void Recomb(Mat &src, Mat &dst)
//{
//	int cx = src.cols >> 1;
//	int cy = src.rows >> 1;
//	Mat tmp;
//	tmp.create(src.size(), src.type());
//	src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
//	src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));
//	src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
//	src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
//	dst = tmp;
//}
////----------------------------------------------------------
//// 2D Forward FFT
////----------------------------------------------------------
//void ForwardFFT(Mat &Src, Mat *FImg, bool do_recomb = true)
//{
//	int M = getOptimalDFTSize(Src.rows);
//	int N = getOptimalDFTSize(Src.cols);
//	Mat padded;
//	copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));
//	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
//	Mat complexImg;
//	merge(planes, 2, complexImg);
//	dft(complexImg, complexImg);
//	split(complexImg, planes);
//	planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
//	planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));
//	if (do_recomb)
//	{
//		Recomb(planes[0], planes[0]);
//		Recomb(planes[1], planes[1]);
//	}
//	planes[0] /= float(M*N);
//	planes[1] /= float(M*N);
//	FImg[0] = planes[0].clone();
//	FImg[1] = planes[1].clone();
//}
////----------------------------------------------------------
//// 2D inverse FFT
////----------------------------------------------------------
//void InverseFFT(Mat *FImg, Mat &Dst, bool do_recomb = true)
//{
//	if (do_recomb)
//	{
//		Recomb(FImg[0], FImg[0]);
//		Recomb(FImg[1], FImg[1]);
//	}
//	Mat complexImg;
//	merge(FImg, 2, complexImg);
//	idft(complexImg, complexImg);
//	split(complexImg, FImg);
//	Dst = FImg[0].clone();
//}
////-----------------------------------------------------------------------------------------------------
////
////-----------------------------------------------------------------------------------------------------
//void highpass(Size sz, Mat& dst)
//{
//	Mat a = Mat(sz.height, 1, CV_32FC1);
//	Mat b = Mat(1, sz.width, CV_32FC1);
//
//	float step_y = CV_PI / sz.height;
//	float val = -CV_PI*0.5;
//
//	for (int i = 0; i < sz.height; ++i)
//	{
//		a.at<float>(i) = cos(val);
//		val += step_y;
//	}
//
//	val = -CV_PI*0.5;
//	float step_x = CV_PI / sz.width;
//	for (int i = 0; i < sz.width; ++i)
//	{
//		b.at<float>(i) = cos(val);
//		val += step_x;
//	}
//
//	Mat tmp = a*b;
//	dst = (1.0 - tmp).mul(2.0 - tmp);
//}
////-----------------------------------------------------------------------------------------------------
////
////-----------------------------------------------------------------------------------------------------
//float logpolar(Mat& src, Mat& dst)
//{
//	float radii = src.cols;
//	float angles = src.rows;
//	Point2f center(src.cols / 2, src.rows / 2);
//	float d = norm(Vec2f(src.cols - center.x, src.rows - center.y));
//	float log_base = pow(10.0, log10(d) / radii);
//	float d_theta = CV_PI / (float)angles;
//	float theta = CV_PI / 2.0;
//	float radius = 0;
//	Mat map_x(src.size(), CV_32FC1);
//	Mat map_y(src.size(), CV_32FC1);
//	for (int i = 0; i < angles; ++i)
//	{
//		for (int j = 0; j < radii; ++j)
//		{
//			radius = pow(log_base, float(j));
//			float x = radius * sin(theta) + center.x;
//			float y = radius * cos(theta) + center.y;
//			map_x.at<float>(i, j) = x;
//			map_y.at<float>(i, j) = y;
//		}
//		theta += d_theta;
//	}
//	remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
//	return log_base;
//}
////-----------------------------------------------------------------------------------------------------
////
////-----------------------------------------------------------------------------------------------------
//RotatedRect imreg(Mat& im0, Mat& im1,Mat &rstImg)
//{
//	Mat F0[2], F1[2];
//	Mat f0, f1;
//	ForwardFFT(im0, F0);
//	ForwardFFT(im1, F1);
//	magnitude(F0[0], F0[1], f0);
//	magnitude(F1[0], F1[1], f1);
//
//	// Create filter 
//	Mat h;
//	highpass(f0.size(), h);
//
//	// Apply it in freq domain
//	f0 = f0.mul(h);
//	f1 = f1.mul(h);
//
//	float log_base;
//	Mat f0lp, f1lp;
//
//	log_base = logpolar(f0, f0lp);
//	log_base = logpolar(f1, f1lp);
//
//	// Find rotation and scale
//	Point2d rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);
//
//	float angle = 180.0 * rotation_and_scale.y / f0lp.rows;
//	float scale = pow(log_base, rotation_and_scale.x);
//	// --------------
//	if (scale > 1.8)
//	{
//		rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);
//		angle = -180.0 * rotation_and_scale.y / f0lp.rows;
//		scale = 1.0 / pow(log_base, rotation_and_scale.x);
//		if (scale > 1.8)
//		{
//			cout << "Images are not compatible. Scale change > 1.8" << endl;
//			return RotatedRect();
//		}
//	}
//	// --------------
//	if (angle < -90.0)
//	{
//		angle += 180.0;
//	}
//	else if (angle > 90.0)
//	{
//		angle -= 180.0;
//	}
//
//	// Now rotate and scale fragmet back, then find translation
//	Mat rot_mat = getRotationMatrix2D(Point(im1.cols / 2, im1.rows / 2), angle, 1.0 / scale);
//
//	// rotate and scale
//	Mat im1_rs;
//	warpAffine(im1, im1_rs, rot_mat, im1.size());
//
//	// find translation
//	Point2d tr = cv::phaseCorrelate(im1_rs, im0);
//
//	// compute rotated rectangle parameters
//	RotatedRect rr;
//	rr.center = tr + Point2d(im0.cols / 2, im0.rows / 2);
//	rr.angle = -angle;
//	rr.size.width = im1.cols / scale;
//	rr.size.height = im1.rows / scale;
//
//	float trans[] = { 1.0 ,0.0, tr.x, 0.0,1.0, tr.y };
//	Mat trans_mat(2, 3, CV_32FC1, trans);
//	Mat rst_Mat;
//	warpAffine(im1_rs, rstImg, trans_mat, im1_rs.size());
//
//
//
//	return rr;
//}
//
//Mat& imregTwice(Mat &im0, Mat &im1, Mat &rstImg)
//{
//	Mat rstImg0;
//	imreg(im0, im1, rstImg0);
//	imreg(im0, rstImg0, rstImg);
//	return rstImg;
//}
//
//int main1(int argc, char* argv[])
//{
//
//	Mat src = imread("lena.jpg", 0);
//
//	Mat src1;
//	src1 = src.clone();
//	medianBlur(src1, src1, 3);
//	src.convertTo(src, CV_32FC1, 1.0 / 244.0);
//	Mat dst = imread("lena2.jpg", 0);
//	dst.convertTo(dst, CV_32FC1, 1.0 / 244.0);
//
//	Mat trans;
//	imreg(src, dst, trans);
//	Mat subImg;
//	absdiff(src, trans, subImg);
//
//	Mat transTwice;
//	imregTwice(src, dst, transTwice);
//	Mat subImgTwice;
//	absdiff(src, transTwice, subImgTwice);
//
//	
//	VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(2*640, 2*480));
//
//
//
//	if (argc != 2)
//	{
//		return 0;
//	}
//	Mat frame0, frame1, frame2, frame3;
//	Mat rstImg0, rstImg1,rstImg2;
//	Mat subImg0, subImg1;
//
//	VideoCapture cap(argv[1]);
//
//
//	//namedWindow("subImg0", 1);
//	//namedWindow("subImg1", 1);
//	namedWindow("moveObject", 1);
//	namedWindow("video", 1);
//	cap >> frame0;
//	cap >> frame1;
//	cap >> frame2;
//
//	cvtColor(frame0, frame0, CV_BGR2GRAY);
//	cvtColor(frame1, frame1, CV_BGR2GRAY);
//	cvtColor(frame2, frame2, CV_BGR2GRAY);
//
//	frame0.convertTo(frame0, CV_32FC1, 1.0 / 255.0);
//	frame1.convertTo(frame1, CV_32FC1, 1.0 / 255.0);
//	frame2.convertTo(frame2, CV_32FC1, 1.0 / 255.0);
//
//	while (waitKey(5) != 'q')
//	{
//		static int i = 0;
//		if (++i == 10)
//		{
//			printf(" \n");
//			//
//		}
//
//		cap >> frame3;
//		if (frame3.empty()) break;
//		imshow("video", frame3);
//
//		cvtColor(frame3, frame3, CV_BGR2GRAY);
//		Mat frame3u = frame3.clone();
//		frame3.convertTo(frame3, CV_32FC1, 1.0 / 255.0);
//		medianBlur(frame3, frame3, 3);
//
//		//imreg(frame0, frame2, rstImg0);
//		//imreg(frame1, frame3, rstImg1);
//
//		imregTwice(frame0, frame2, rstImg0);
//		imregTwice(frame1, frame3, rstImg1);
//
//		//medianBlur(frame0, frame0, 3);
//		//medianBlur(frame1, frame1, 3);
//
//		//GaussianBlur(frame0, frame0, Size(3, 3), 0.5);
//		GaussianBlur(frame1, frame1, Size(3, 3), 0.5);
//		absdiff(frame0, rstImg0, subImg0);
//		absdiff(frame1, rstImg1, subImg1);
//
//		Mat result = subImg1.mul(subImg0);
//		
//		//imshow("subImg0", subImg0);
//		//imshow("subImg1", subImg1);
//		//imshow("moveObject", result);
//
//		Mat normResult;
//		normalize(result, normResult, 255.0, 0.0, NORM_MINMAX);
//		normResult.convertTo(normResult, CV_8UC1);
//
//
//		// threshold
//		Mat thresImg;
//		threshold(normResult, thresImg, 20, 255, THRESH_BINARY);
//
//		Mat erodeImg, dilateImg;
//		Mat kernelErode(2, 2, CV_8UC1, Scalar(1));
//		Mat kernelDilate(5, 5, CV_8UC1, Scalar(1));
//
//		erode(thresImg, erodeImg, kernelErode);
//		dilate(erodeImg, dilateImg, kernelDilate);
//
//		
//
//		Mat VecReduce,vecNorm;
//		Mat histImg(480, 640, CV_32FC1, Scalar(125));
//		reduce(dilateImg, VecReduce, 0, CV_REDUCE_SUM, CV_MAT_DEPTH(CV_32F));
//
//		normalize(VecReduce, VecReduce,480,0,NORM_MINMAX);
//		for (int i = 0 ; i < 640; ++i)
//		{
//			int h = (int)VecReduce.at<float>(0, i);
//			histImg(Rect(i, 0, 1, h)) = Scalar(255);
//		}
//		flip(histImg, histImg, 0);
//		histImg.convertTo(histImg, CV_8UC1);
//
//		// 保存记录
//		Mat saveImg(2*480, 640*2, CV_8UC1,Scalar(0));
//		dilateImg.copyTo(saveImg(Rect(0, 0, 640, 480)));
//		frame3u.copyTo(saveImg(Rect(640, 0, 640, 480)));
//		histImg.copyTo(saveImg(Rect(0, 480, 640, 480)));
//	
//		imshow("moveObject", saveImg);
//		writer << saveImg;
//		frame0 = frame1;
//		frame1 = frame2;
//		frame2 = frame3;
//	}
//	writer.release();
//	return 0;
//}
//
////#include<iostream>
////#include<opencv2/opencv.hpp>
////using namespace std;
////
////int main1()
////{
////	//数组声明
////	CvPoint2D32f srcTri[3], dstTri[3];
////	//创建指针
////	CvMat* warp_mat = cvCreateMat(2, 3, CV_32FC1);
////	CvMat* rot_mat = cvCreateMat(2, 3, CV_32FC1);
////	//载入和显示图像
////	IplImage *src;
////	src = cvLoadImage("lena.jpg", CV_LOAD_IMAGE_UNCHANGED);
////	cvNamedWindow("原图", CV_WINDOW_AUTOSIZE);
////	cvShowImage("原图", src);
////
////	IplImage *dst = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
////	dst = cvCloneImage(src);
////
////	dst->origin = src->origin;
////	cvZero(dst);
////	//计算变换矩阵
////	srcTri[0].x = 0;
////	srcTri[0].y = 0;
////	srcTri[1].x = src->width - 1;
////	srcTri[1].y = 0;
////	srcTri[2].x = 0;
////	srcTri[2].y = src->height - 1;
////
////	dstTri[0].x = src->width*0.0;
////	dstTri[0].y = src->height*0.33;
////	dstTri[1].x = src->width*0.85;
////	dstTri[1].y = src->height*0.25;
////	dstTri[2].x = src->width*0.15;
////	dstTri[2].y = src->height*0.7;
////
////	cvGetAffineTransform(srcTri, dstTri, warp_mat);
////	//调用函数cvWarpAffine（）
////	cvWarpAffine(src, dst, warp_mat);
////	cvNamedWindow("仿射图1", CV_WINDOW_AUTOSIZE);
////	cvShowImage("仿射图1", dst);
////
////	cvCopy(dst, src);
////	//用另外一种方法得到变换矩阵，并进行仿射变换
////	CvPoint2D32f center = cvPoint2D32f(src->height / 2, src->width / 2);
////	double angle = -2.0;
////	double scale = 0.96;
////	cv2DRotationMatrix(center, angle, scale, rot_mat);
////
////	//IplImage * src1 = cvLoadImage("lena.jpg", CV_LOAD_IMAGE_UNCHANGED);
////	//cvWarpAffine(src1, dst, rot_mat);
////
////
////	cv::Mat src1, dst1;
////	src1 = imread("lena.jpg", 1);
////	Mat tran = cv::getRotationMatrix2D(center, angle, scale);
////
////	cv::warpAffine(src1, dst1, tran,cvSize(512,512));
////	
////	cvNamedWindow("仿射图2", CV_WINDOW_AUTOSIZE);
////	imshow("仿射图2", dst1);
////	imwrite("lena2.jpg", dst1);
////
////	cvWaitKey();
////	cvReleaseImage(&src);
////	cvReleaseImage(&dst);
////	cvDestroyWindow("原图");
////	cvDestroyWindow("仿射图1");
////	cvDestroyWindow("仿射图2");
////	cvReleaseMat(&rot_mat);
////	cvReleaseMat(&warp_mat);
////
////	return 0;
////
////
////}
