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

Mat image;
Mat gray;

int main(){
    image=imread("/Users/wellcome/Desktop/PDI_LISTA/image.png",1);
    
    cvtColor(image, gray, CV_RGB2GRAY);
    namedWindow("Image Gray",CV_WINDOW_AUTOSIZE);
    imshow("Image Gray", gray);
    
    Mat_<Vec3b> img(gray.rows,gray.cols,CV_8UC3);
    
    for(int x=0;x<gray.rows;x++)
    {
        for(int y=0;y<gray.cols;y++)
        {
            if((x==gray.rows/2)&&(y==gray.cols/2))
            {
                img(x,y)[0]=0;
                img(x,y)[1]=255;
                img(x,y)[2]=0;
            }
            else
            {
                img(x,y)[0]=gray.at<uchar>(y,x);
                img(x,y)[1]=gray.at<uchar>(y,x);
                img(x,y)[2]=gray.at<uchar>(y,x);
            }
        }
    }
    
    namedWindow("Centro",CV_WINDOW_AUTOSIZE);
    imshow("Centro",img);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Centro.jpg",img);
    
    waitKey();
    
}