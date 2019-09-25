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

int main(){
    Mat img, gray;
    
    img = imread("/Users/wellcome/Desktop/PDI_LISTA/img.png", 1 );
    cvtColor( img, gray, COLOR_BGR2GRAY );
    
    //reduzir ruidos na imagem
    GaussianBlur(gray, gray, Size(9, 9), 2, 2 );
    
    vector <Vec3f> circles;
    HoughCircles (gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows/8, 200, 100, 0, 0.);
    
    for(int i=0; i<circles.size(); i++){
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        
        int radius = cvRound(circles[i][2]);
        
        circle( img, center, 3, Scalar(0,255,0), -1, 8, 0 );
        
        circle( img, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }
    imshow("Hough Transform", img);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Hough.jpg", img);
    waitKey();
}