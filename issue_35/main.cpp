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
    Mat image=imread("/Users/wellcome/Desktop/PDI_LISTA/img.png",1);
    Mat gray,limi;
    
    cvtColor(image, gray, CV_RGB2GRAY);
    imshow("Gray Image", gray);
    waitKey();
    
    threshold(gray, limi, 127, 255, CV_THRESH_OTSU);
    imshow("Threshold_Otsu Image", limi);
    waitKey();
    
    for (int x=0; x<80; x++){
        dilate(limi, limi, getStructuringElement(CV_SHAPE_RECT, Size(3,3)));
        imshow("Dilate", limi);
        waitKey(30);
    }
}