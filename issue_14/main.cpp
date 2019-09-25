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

Mat gray;
Mat video;

int main(){
    VideoCapture captura(0);
    while(1){
        captura >> video;
        namedWindow("Video", CV_WINDOW_AUTOSIZE);
        cvtColor(video, gray, CV_RGB2GRAY);
        imshow("Video", gray);
    }
}