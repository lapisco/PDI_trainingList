#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

Mat video;
Mat img;

int main(){
    VideoCapture capture(0);
    while(1){
        capture>>video;
        cvtColor(video, img, CV_RGB2GRAY);
        Laplacian(img, img, CV_CANNY_L2_GRADIENT);
        imshow("Image Canny", img);
    }
}