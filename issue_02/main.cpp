#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;


Mat img_rgb;
Mat img_gray;

int main()
{
    img_rgb=imread("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/arara.jpg",1);
    
    namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    imshow("Original Image", img_rgb);
    
    cvtColor(img_rgb, img_gray, CV_RGB2GRAY);
    
    namedWindow("Image Gray", CV_WINDOW_AUTOSIZE);
    imshow("Image Gray", img_gray);
    
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Floresta_Gray.jpg", img_gray);
    
    waitKey(0);
    
}

