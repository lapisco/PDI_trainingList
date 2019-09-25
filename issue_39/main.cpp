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
    Mat image=imread("/Users/wellcome/Desktop/PDI_LISTA/img.jpg",1);
    Mat gray,limi;
    
    cvtColor(image, gray, CV_RGB2GRAY);
    imshow("Gray Image", gray);
    waitKey();
    
    threshold(gray, limi, 127, 255, CV_THRESH_OTSU);
    imshow("Threshold_Otsu Image", limi);
    waitKey();
    
    Mat element = getStructuringElement(CV_SHAPE_RECT, Size(1,3));
    
    for (int x=0; x<80; x++){
        erode(limi, limi, element);
        imshow("Erode", limi);
        waitKey(30);
    }
}