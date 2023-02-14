#pragma once

#include "opencv2/opencv.hpp"

/*
calibrationdata_20230202_1920x1080.txt
*/

//ds objectives

double camera_calibration_matrix_left_data[] = {1112.826589, 0.000000, 975.219381,
                                           0.000000, 1109.747216, 526.077095,
                                           0.000000, 0.000000, 1.000000};
double camera_calibration_matrix_right_data[] = {1131.184430, 0.000000, 930.237238,
                                            0.000000, 1123.173525, 523.001003,
                                            0.000000, 0.000000, 1.000000};
double distortion_coefficients_left_data[] = {-0.328047, 0.096274, 0.001224, -0.001850, 0.000000};
double distortion_coefficients_right_data[] = {-0.331597, 0.106119, 0.003659, -0.003671, 0.000000};
double rectification_matrix_left_data[] = {0.999912, -0.013237, -0.001307,
                                      0.013243, 0.999902, 0.004510,
                                      0.001247, -0.004527, 0.999989};
double rectification_matrix_right_data[] = {0.997799, -0.010548, -0.065460,
                                       0.010252, 0.999936, -0.004857,
                                       0.065507, 0.004175, 0.997843};
double projection_matrix_left_data[] = {1041.789380, 0.000000, 1001.367607, 0.000000,
                                   0.000000, 1041.789380, 532.560921, 0.000000,
                                   0.000000, 0.000000, 1.000000, 0.000000};
double projection_matrix_right_data[] = {1041.789380, 0.000000, 1001.367607, -1119.920486,
                                    0.000000, 1041.789380, 532.560921, 0.000000,
                                    0.000000, 0.000000, 1.000000, 0.000000};

cv::Mat camera_calibration_matrix_left(cv::Size(3, 3), CV_64FC1, camera_calibration_matrix_left_data);
cv::Mat camera_calibration_matrix_right(cv::Size(3, 3), CV_64FC1, camera_calibration_matrix_right_data);
cv::Mat distortion_coefficients_left(cv::Size(1, 5), CV_64FC1, distortion_coefficients_left_data);
cv::Mat distortion_coefficients_right(cv::Size(1, 5), CV_64FC1, distortion_coefficients_right_data);
cv::Mat projection_matrix_left(cv::Size(4, 3), CV_64FC1, projection_matrix_left_data);
cv::Mat projection_matrix_right(cv::Size(4, 3), CV_64FC1, projection_matrix_right_data);
cv::Mat rectification_matrix_left(cv::Size(3, 3), CV_64FC1, rectification_matrix_left_data);
cv::Mat rectification_matrix_right(cv::Size(3, 3), CV_64FC1, rectification_matrix_right_data);