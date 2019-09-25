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

FILE*img;
Mat image;
Mat gray;

int main()
{
    image=imread("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/paisagem.jpg",1);
    namedWindow("Original Image",CV_WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    
    cvtColor(image, gray, CV_RGB2GRAY);
    namedWindow("Image Gray",CV_WINDOW_AUTOSIZE);
    imshow("Image Gray", gray);
    
    int matriz[320][240];
    
    for (int x=0; x<gray.cols; x++) {
        for (int y=0; y<gray.rows; y++) {
            matriz[x][y]=gray.at<Vec3b>(x,y)[0];
        }
    }
    img=fopen("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/image.txt", "w");
    for (int x=0; x<gray.cols; x++) {
        for (int y=0; y<gray.rows; y++) {
            fprintf(img, "%d ",matriz[x][y]);
        }
        fprintf(img, "\n");
    }
    fclose(img);
    waitKey(0);
    
}