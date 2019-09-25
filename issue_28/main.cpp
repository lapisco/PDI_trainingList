#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

int main(){
    Mat image = imread("/Users/wellcome/Desktop/PDI_LISTA/arara.jpg",1);
    Mat gray,limi;
    cvtColor(image, gray, CV_RGB2GRAY);
    adaptiveThreshold(gray, limi, 227, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 5, -20);
    imshow("Image Thresholded", limi);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/thresholded.jpg", limi);
    
    waitKey(0);
}
