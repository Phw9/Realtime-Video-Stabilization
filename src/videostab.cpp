#include <cmath>
#include "videostab.h"
#include "opencv2/xfeatures2d.hpp"

#define MAXCORNERS 200  // 1000
#define MINDISTANCE 20  // 20

//Parameters for Kalman Filter
#define Q1 0.004    // 0.004
#define R1 0.5      // 0.5


//To see the results of before and after stabilization simultaneously
#define test 1

VideoStab::VideoStab()
{

    smoothedMat.create(2 , 3 , CV_64F);

    k = 1;

    errscaleX = 1;
    errscaleY = 1;
    errthetha = 1;
    errtransX = 1;
    errtransY = 1;

    Q_scaleX = Q1;
    Q_scaleY = Q1;
    Q_thetha = Q1;
    Q_transX = Q1;
    Q_transY = Q1;

    R_scaleX = R1;
    R_scaleY = R1;
    R_thetha = R1;
    R_transX = R1;
    R_transY = R1;

    sum_scaleX = 0;
    sum_scaleY = 0;
    sum_thetha = 0;
    sum_transX = 0;
    sum_transY = 0;

    scaleX = 0;
    scaleY = 0;
    thetha = 0;
    transX = 0;
    transY = 0;

}

//The main stabilization function
cv::Mat VideoStab::stabilize(cv::Mat frame_1, cv::Mat frame_2)
{
    cv::cvtColor(frame_1, frame1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame_2, frame2, cv::COLOR_BGR2GRAY);
    int vert_border = HORIZONTAL_BORDER_CROP * frame_1.rows / frame_1.cols;
    std::vector <cv::Point2f> features1, features2;
    std::vector <cv::Point2f> goodFeatures1, goodFeatures2;
    std::vector <uchar> status;
    std::vector <float> err;

    //Estimating the features in frame1 and frame2
    cv::goodFeaturesToTrack(frame1, features1, MAXCORNERS, 0.01, MINDISTANCE);
    
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief1 = 
                        cv::xfeatures2d::BriefDescriptorExtractor::create();
    std::vector<cv::KeyPoint> kp1;
    std::vector<uchar> idx1; cv::Mat desc1;
    std::vector<std::vector<DTYPE>> vdesc1;
    VecToKeyPoint(features1, kp1);
    brief1->compute(frame1, kp1, desc1);
    features1.clear();  KeyPointToVec(kp1, features1);  
    vdesc1.clear();  MatToVec(desc1, vdesc1);

    cv::calcOpticalFlowPyrLK(frame1, frame2, features1, features2, status, err);
    
    int indexCorrection = 0;
    int N = status.size();
    for(int i = 0; i < N; i++)
    {
        cv::Point2f pt = features2.at(i - indexCorrection);

        if((status.at(i) == 0) || 
            (pt.x < 0 || pt.x > (float)1920.0) || 
            (pt.y < 0 || pt.y > (float)1080.0) ||
            err.at(i) > 20)
        {
            if((pt.x < 0 || pt.x > (float)1920.0) || 
                (pt.y < 0 || pt.y > (float)1080.0) ||
                err.at(i) > 1)
            {
                status.at(i) = 0;
            }
            features1.erase(features1.begin() + (i - indexCorrection));
            features2.erase(features2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }

    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief2 = 
                cv::xfeatures2d::BriefDescriptorExtractor::create();
    std::vector<cv::KeyPoint> kp2;
    std::vector<uchar> idx2; cv::Mat desc2;
    std::vector<std::vector<DTYPE>> vdesc2;
    std::vector<uchar> delete2;
    VecToKeyPoint(features2, kp2);
    brief2->compute(frame2, kp2, desc2);
    delete2 = FindDeletePoints(kp2, features2);
    DeletePoints(delete2, vdesc1, features1);
    features2.clear(); vdesc2.clear();
    KeyPointToVec(kp2, features2);
    MatToVec(desc2, vdesc2);

    // // disc matching
    // cv::BFMatcher matcher(cv::NORM_HAMMING);
    // std::vector<cv::DMatch> matches;
    // matcher.match(desc1, desc2, matches);
    // int M = features1.size();
    // std::pair<int,int> idxMatch;
    // std::vector<std::pair<int,int>> temp;
    // int indexCorrection1 = 0;

    // for(int i = 0; i < M; i++)
    // {
    //     if(matches.at(i).distance < 10)
    //     {
    //         if(matches.at(i).queryIdx != matches.at(i).trainIdx)
    //         {
    //             features1.erase(features1.begin() + (i - indexCorrection));
    //             features2.erase(features2.begin() + (i - indexCorrection));
    //             indexCorrection++;
    //         }
    //     }
    //     // else
    //     // {
    //     //     features1.erase(features1.begin() + (i - indexCorrection));
    //     //     features2.erase(features2.begin() + (i - indexCorrection));
    //     //     indexCorrection++;
    //     // }
    // }

    for(size_t i = 0; i < features1.size(); i++)
    {
        goodFeatures1.push_back(features1[i]);
        goodFeatures2.push_back(features2[i]);
    }

    //All the parameters scale, angle, and translation are stored in affine
    affine = cv::estimateRigidTransform(goodFeatures1, goodFeatures2, false);
    if(affine.rows == 0 || affine.cols == 0) return frame_2;
    // std::cout << affine << std::endl;
    std::flush(std::cout);
    
    //affine = affineTransform(goodFeatures1 , goodFeatures2);

    dx = affine.at<double>(0,2);
    dy = affine.at<double>(1,2);
    da = atan2(affine.at<double>(1,0), affine.at<double>(0,0));
    ds_x = affine.at<double>(0,0)/cos(da);
    ds_y = affine.at<double>(1,1)/cos(da);

    sx = ds_x;
    sy = ds_y;

    sum_transX += dx;
    sum_transY += dy;
    sum_thetha += da;
    sum_scaleX += ds_x;
    sum_scaleY += ds_y;


    //Don't calculate the predicted state of Kalman Filter on 1st iteration
    if(k==1)
    {
        k++;
    }
    else
    {
        Kalman_Filter(&scaleX , &scaleY , &thetha , &transX , &transY);

    }

    diff_scaleX = scaleX - sum_scaleX;
    diff_scaleY = scaleY - sum_scaleY;
    diff_transX = transX - sum_transX;
    diff_transY = transY - sum_transY;
    diff_thetha = thetha - sum_thetha;

    ds_x = ds_x + diff_scaleX;
    ds_y = ds_y + diff_scaleY;
    dx = dx + diff_transX;
    dy = dy + diff_transY;
    da = da + diff_thetha;

    //Creating the smoothed parameters matrix
    smoothedMat.at<double>(0,0) = sx * cos(da);
    smoothedMat.at<double>(0,1) = sx * -sin(da);
    smoothedMat.at<double>(1,0) = sy * sin(da);
    smoothedMat.at<double>(1,1) = sy * cos(da);

    smoothedMat.at<double>(0,2) = dx;
    smoothedMat.at<double>(1,2) = dy;

    //Uncomment if you want to see smoothed values
    //cout<<smoothedMat;
    //flush(cout);

    //Warp the new frame using the smoothed parameters
    cv::warpAffine(frame_1, smoothedFrame, smoothedMat, frame_2.size());

    //Crop the smoothed frame a little to eliminate black region due to Kalman Filter
    smoothedFrame = smoothedFrame(cv::Range(vert_border, smoothedFrame.rows-vert_border), cv::Range(HORIZONTAL_BORDER_CROP, smoothedFrame.cols-HORIZONTAL_BORDER_CROP));

    cv::resize(smoothedFrame, smoothedFrame, frame_2.size());

    //Change the value of test if you want to see both unstabilized and stabilized video
    if(test)
    {
        cv::Mat canvas = cv::Mat::zeros(frame_2.rows, frame_2.cols*2+10, frame_2.type());
        cv::Mat graycanvas = cv::Mat::zeros(frame2.rows, frame2.cols, frame2.type());
        frame2.copyTo(graycanvas(cv::Range::all(), cv::Range(0, graycanvas.cols)));

        frame_1.copyTo(canvas(cv::Range::all(), cv::Range(0, smoothedFrame.cols)));

        smoothedFrame.copyTo(canvas(cv::Range::all(), cv::Range(smoothedFrame.cols+10, smoothedFrame.cols*2+10)));

        cv::Mat crop1 = frame_1(cv::Rect(frame_1.cols/4, frame_1.rows/4, frame_1.cols/1.5, frame_1.rows/1.5));
        cv::Mat crop2 = smoothedFrame(cv::Rect(smoothedFrame.cols/4, smoothedFrame.rows/4, smoothedFrame.cols/1.5, smoothedFrame.rows/1.5));
        cv::Mat canvas1 = cv::Mat::zeros(crop1.rows, crop1.cols*2+10, crop1.type());

        crop1.copyTo(canvas1(cv::Range::all(), cv::Range(0, crop2.cols)));

        crop2.copyTo(canvas1(cv::Range::all(), cv::Range(crop2.cols+10, crop2.cols*2+10)));
       
        DrawFeatures(graycanvas, goodFeatures1, goodFeatures2);
        
        if(canvas.cols > 1920)
        {
            cv::resize(canvas, canvas, cv::Size(canvas.cols/2, canvas.rows/2));
            cv::resize(graycanvas, graycanvas, cv::Size(graycanvas.cols/2, graycanvas.rows/2));
        }

        if(canvas1.cols > 1920)
        {
            cv::resize(canvas1, canvas1, cv::Size(canvas.cols/2, canvas.rows/2));
        }

        std::cout << "Feature size: " << goodFeatures1.size() << ", ";

        cv::imshow("before and after", canvas);
        cv::imshow("before and after crop", canvas1);
        cv::imshow("grayscale", graycanvas);

        cv::Mat hist1, hist2;
        int channels[] = {0};
        int histSize[] = {256};
        float range[] = {0, 256};
        const float* ranges[] = {range};
        cv::calcHist(&frame_1, 1, channels, cv::Mat(), hist1, 1, histSize, ranges);
        cv::calcHist(&frame_2, 1, channels, cv::Mat(), hist2, 1, histSize, ranges);
        cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        double hist_score = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
        
        hist1 = hist1.t(); hist2 = hist2.t();
        std::cout << hist1.size()<<", " << hist1.rows << ", " <<hist1.cols << std::endl;
        cv::Mat hist_canvas = cv::Mat::zeros(hist1.rows+4, hist1.cols, hist1.type());
        hist1.copyTo(hist_canvas(cv::Range(0, hist1.rows), cv::Range::all()));
        hist2.copyTo(hist_canvas(cv::Range(hist2.rows+2, hist2.rows*2+2), cv::Range::all()));
        cv::resize(hist_canvas, hist_canvas, cv::Size(hist_canvas.cols*3, hist_canvas.rows*100));
        
        cv::imshow("histogram", hist_canvas);
        
        std::cout << "hist_score : " << hist_score << std::endl; 
    }
    return smoothedFrame;

}


//Kalman Filter implementation
void VideoStab::Kalman_Filter(double *scaleX , double *scaleY , double *thetha , double *transX , double *transY)
{
    double frame_1_scaleX = *scaleX;
    double frame_1_scaleY = *scaleY;
    double frame_1_thetha = *thetha;
    double frame_1_transX = *transX;
    double frame_1_transY = *transY;

    double frame_1_errscaleX = errscaleX + Q_scaleX;
    double frame_1_errscaleY = errscaleY + Q_scaleY;
    double frame_1_errthetha = errthetha + Q_thetha;
    double frame_1_errtransX = errtransX + Q_transX;
    double frame_1_errtransY = errtransY + Q_transY;

    double gain_scaleX = frame_1_errscaleX / (frame_1_errscaleX + R_scaleX);
    double gain_scaleY = frame_1_errscaleY / (frame_1_errscaleY + R_scaleY);
    double gain_thetha = frame_1_errthetha / (frame_1_errthetha + R_thetha);
    double gain_transX = frame_1_errtransX / (frame_1_errtransX + R_transX);
    double gain_transY = frame_1_errtransY / (frame_1_errtransY + R_transY);

    *scaleX = frame_1_scaleX + gain_scaleX * (sum_scaleX - frame_1_scaleX);
    *scaleY = frame_1_scaleY + gain_scaleY * (sum_scaleY - frame_1_scaleY);
    *thetha = frame_1_thetha + gain_thetha * (sum_thetha - frame_1_thetha);
    *transX = frame_1_transX + gain_transX * (sum_transX - frame_1_transX);
    *transY = frame_1_transY + gain_transY * (sum_transY - frame_1_transY);

    errscaleX = ( 1 - gain_scaleX ) * frame_1_errscaleX;
    errscaleY = ( 1 - gain_scaleY ) * frame_1_errscaleX;
    errthetha = ( 1 - gain_thetha ) * frame_1_errthetha;
    errtransX = ( 1 - gain_transX ) * frame_1_errtransX;
    errtransY = ( 1 - gain_transY ) * frame_1_errtransY;
}

