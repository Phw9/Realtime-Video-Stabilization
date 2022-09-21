#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include <videostab.h>

using namespace std;
using namespace cv;

// This class redirects cv::Exception to our process so that we can catch it and handle it accordingly.
class cvErrorRedirector {
public:
    int cvCustomErrorCallback( )
    {
        std::cout << "A cv::Exception has been caught. Skipping this frame..." << std::endl;
        return 0;
    }

    cvErrorRedirector() {
        cvRedirectError((cv::ErrorCallback)cvCustomErrorCallback(), this);
    }
};

const int HORIZONTAL_BORDER_CROP = 30;

int main(int argc, char **argv)
{
    cvErrorRedirector redir;
    
    //Create a object of stabilization class
    VideoStab stab;

    //Initialize the VideoCapture object
    VideoCapture cap("video_8floor.mp4");

    Mat frame_2, frame2;
    Mat frame_1, frame1;

    cap >> frame_1;
    cvtColor(frame_1, frame1, COLOR_BGR2GRAY);

    Mat smoothedMat(2,3,CV_64F);

    VideoWriter outputVideo;
    outputVideo.open("com.avi" , CV_FOURCC('X' , 'V' , 'I' , 'D'), 30 , frame_1.size());

    while(true)
    {
        try {
            cap >> frame_2;

            if(frame_2.data == NULL)
            {
                break;
            }

            cvtColor(frame_2, frame2, COLOR_BGR2GRAY);

            Mat smoothedFrame;

            smoothedFrame = stab.stabilize(frame_1 , frame_2);

            outputVideo.write(smoothedFrame);

            imshow("Stabilized Video" , smoothedFrame);

            waitKey(10);

            frame_1 = frame_2.clone();
            frame2.copyTo(frame1);
        } catch (cv::Exception& e) {
            cap >> frame_1;
            cvtColor(frame_1, frame1, COLOR_BGR2GRAY);
        }

    }

    return 0;
}


