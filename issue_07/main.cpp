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


Mat image_rgb;
Mat image_gray;
Mat image_limi;

int main()
{
    image_rgb=imread("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/arara.jpg",1);
    namedWindow("Original Image",CV_WINDOW_AUTOSIZE);
    imshow("Original Image", image_rgb);
    
    cvtColor(image_rgb, image_gray, CV_RGB2GRAY);
    namedWindow("Image Gray",CV_WINDOW_AUTOSIZE);
    imshow("Image Gray", image_gray);
    
    threshold(image_gray, image_limi, 100, 200, CV_THRESH_BINARY);
    namedWindow("Image Thresholded",CV_WINDOW_AUTOSIZE);
    imshow("Image Thresholded", image_limi);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Image_Thresholded.jpg", image_limi);
    
    cvWaitKey(0);
    
}

