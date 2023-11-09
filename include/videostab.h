#pragma once

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#define DTYPE uchar

class VideoStab
{
public:
    VideoStab();
    VideoStab(const std::string& videoName);
    cv::VideoCapture capture;
    std::string videoName_;
    cv::Mat frame2;
    cv::Mat frame1;

    int k;

    const int HORIZONTAL_BORDER_CROP = 30;

    cv::Mat smoothedMat;
    cv::Mat affine;

    cv::Mat smoothedFrame;

    double dx ;
    double dy ;
    double da ;
    double ds_x ;
    double ds_y ;

    double sx ;
    double sy ;

    double scaleX ;
    double scaleY ;
    double thetha ;
    double transX ;
    double transY ;

    double diff_scaleX ;
    double diff_scaleY ;
    double diff_transX ;
    double diff_transY ;
    double diff_thetha ;

    double errscaleX ;
    double errscaleY ;
    double errthetha ;
    double errtransX ;
    double errtransY ;

    double Q_scaleX ;
    double Q_scaleY ;
    double Q_thetha ;
    double Q_transX ;
    double Q_transY ;

    double R_scaleX ;
    double R_scaleY ;
    double R_thetha ;
    double R_transX ;
    double R_transY ;

    double sum_scaleX ;
    double sum_scaleY ;
    double sum_thetha ;
    double sum_transX ;
    double sum_transY ;

    cv::Mat stabilize(cv::Mat frame_1 , cv::Mat frame_2, const std::string& videoName);
    void Kalman_Filter(double *scaleX , double *scaleY , double *thetha , double *transX , double *transY);
};

bool VecToKeyPoint(const std::vector<cv::Point2f>& features2d, std::vector<cv::KeyPoint>& kp)
{
    cv::KeyPoint temp;
    int N = features2d.size();
    for(int i = 0; i < N; i++)
    {
        temp.pt.x = features2d.at(i).x;
        temp.pt.y = features2d.at(i).y;
        kp.emplace_back(std::move(temp));
    }

    if(features2d.size() != kp.size()) return false;

    return true;
}

bool KeyPointToVec(const std::vector<cv::KeyPoint>& kp, std::vector<cv::Point2f>& features2d)
{
    cv::Point2f temp;
    int N = kp.size();
    for(int i = 0; i < N; i++)
    {
        temp.x = kp.at(i).pt.x;
        temp.y = kp.at(i).pt.y;
        features2d.emplace_back(std::move(temp));
    }

    if(features2d.size() != kp.size()) return false;

    return true;
}

bool MatToVec(const cv::Mat& m, std::vector<std::vector<DTYPE>>& v)
{
    std::vector<DTYPE> temp;
    for(int j=0; j<m.rows ; j++)
    {
        for(int i=0; i<m.cols; i++)
        {
            temp.emplace_back(m.at<uchar>(j, i));
        }
        v.emplace_back(std::move(temp));
        temp.clear();
    }
    int N = v.size();
    if(m.rows != N) return false;

    return true;
}

std::vector<uchar> FindDeletePoints(std::vector<cv::KeyPoint>& kp, std::vector<cv::Point2f>& mfeatures)
{
    // kp < mfeatures
    std::vector<uchar> index;
    int k = 0;
    int M = mfeatures.size();
    int P = kp.size();
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < P; j++)
        {
            if(mfeatures.at(i).x == kp.at(j).pt.x && mfeatures.at(i).y == kp.at(j).pt.y)
            {
                index.emplace_back(1);
                k = 1;
                break;
            }
        }
        if(k == 0)
        {
            index.emplace_back(0);
        }
        k = 0;
    }
    int t = 0;
    int f = 0;
    int N = index.size();
    for(int i = 0; i < N; i++)
    {
        if(index.at(i) == 0) f++;
        else t++;
    }
    return index;
}

void DeletePoints(std::vector<uchar>& idx, std::vector<std::vector<uchar>>& mvdesc, std::vector<cv::Point2f>& mfeatures)
{
    if(idx.size() != mfeatures.size())
    {
        std::cerr <<"different number of DeletePoints" << std::endl;
        return;
    }

    int indexCorrection = 0;
    int N = idx.size();
    for(int i = 0; i < N; i++)
    {
        if(idx.at(i) == 0)
        {
            mfeatures.erase(mfeatures.begin() + (i-indexCorrection));
            mvdesc.erase(mvdesc.begin() + (i-indexCorrection));
            indexCorrection++;
        }
    }
}

cv::Mat DrawFeatures(cv::Mat& src, 
                    std::vector<cv::Point2f>& beforePoints, 
                    std::vector<cv::Point2f>& afterPoints)
{
    int N = beforePoints.size();
    cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < N; i++)
	{
        // afterPoints.at(i).x = afterPoints.at(i).x + src.cols/2;
        //random color
        int rgb[3];
        rgb[0]=rand()%256;
        rgb[1]=rand()%256;
        rgb[2]=rand()%256;
        cv::line(src, beforePoints[i], afterPoints[i],cv::Scalar(rgb[0],rgb[1],rgb[2]),2,8,0);
        cv::circle(src, beforePoints[i], 10, cv::Scalar(rgb[0], rgb[1], rgb[2]), 1, 8, 0); //2d features  
        // circle(src, afterPoints[i], 6, Scalar(rgb[0], rgb[1], rgb[2]), 1, 8, 0); //3d points
        cv::rectangle(src, cv::Rect(cv::Point(afterPoints[i].x-10,afterPoints[i].y-10),
        cv::Point(afterPoints[i].x+10,afterPoints[i].y+10)), cv::Scalar(rgb[0],rgb[1],rgb[2]),2,8,0);//projection 3d points
    }
    return src;    
}

