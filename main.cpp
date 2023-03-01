#include <fstream>
#include <time.h>

#include "calibration.h"
#include "videostab.h"

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
    cv::VideoCapture cap("/data/stabil/test/20230130/9/test_cam1.mkv");
    cv::VideoCapture cap2("/data/stabil/test/20230130/9/test_cam2.mkv");

    cv::Mat frame_1, frame1;
    cv::Mat frame_2, frame2;

    cap >> frame_1;
    // intrinsic calib
    cv::cvtColor(frame_1, frame1, cv::COLOR_BGR2GRAY);
    frame_1 = cv::getOptimalNewCameraMatrix(camera_calibration_matrix_left, distortion_coefficients_left, frame_1.size(), 0);

    cv::Mat smoothedMat(2, 3, CV_64F);

    cv::VideoWriter outputVideo;
    outputVideo.open("com.avi" , cv::VideoWriter::fourcc('X' , 'V' , 'I' , 'D'), 30 , frame_1.size());

    while(true)
    {
        try {
            cap >> frame_2;

            if(frame_2.data == NULL)
            {
                break;
            }

            cv::Mat smoothedFrame;

            smoothedFrame = stab.stabilize(frame_1, frame_2);

            outputVideo.write(smoothedFrame);

            // imshow("Stabilized Video" , smoothedFrame);

            char ch = cv::waitKey(33);
            if(ch == 27) break; // ESC key
		    if(ch == 32) if(cv::waitKey(0) == 27) break;; // Spacebar key

            frame_1 = frame_2.clone();
            frame2.copyTo(frame1);
        } catch (cv::Exception& e) {
            cap >> frame_1;
            cv::cvtColor(frame_1, frame1, cv::COLOR_BGR2GRAY);
        }

    }

    return 0;
}


