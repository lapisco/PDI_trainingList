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
    int x,y;
    Mat image=imread("/Users/wellcome/Desktop/PDI_LISTA/figura.jpg",0);
    Mat grow(image.rows,image.cols,CV_8UC1);
    
    for (x=0; x<grow.rows; x++) {
        for (y=0; y<grow.cols; y++) {
            if (x==grow.rows/2 && y==grow.cols/2){
                grow.at<uchar>(x,y)=255;
            }
            else{
                grow.at<uchar>(x,y)=0;
            }
        }
    }
    int parar=0;
    int continuar=1;
    while (continuar!=parar) {
        continuar=parar;
        parar=0;
        for (x=0; x<grow.rows; x++) {
            for (y=0; y<grow.cols; y++) {
                if (grow.at<uchar>(x,y)==255){
                    if (image.at<uchar>(x-1,y-1)<127){
                        grow.at<uchar>(x-1,y-1)=255;
                        parar++;
                    }
                    if (image.at<uchar>(x-1,y)<127){
                        grow.at<uchar>(x-1,y)=255;
                        parar++;
                    }
                    if (image.at<uchar>(x-1,y+1)<127){
                        grow.at<uchar>(x-1,y+1)=255;
                        parar++;
                    }
                    if (image.at<uchar>(x,y-1)<127){
                        grow.at<uchar>(x,y-1)=255;
                        parar++;
                    }
                    if (image.at<uchar>(x,y+1)<127){
                        grow.at<uchar>(x,y+1)=255;
                        parar++;
                    }
                    if (image.at<uchar>(x+1,y-1)<127){
                        grow.at<uchar>(x+1,y-1)=255;
                        parar++;
                    }
                    if (image.at<uchar>(x+1,y)<127){
                        grow.at<uchar>(x+1,y)=255;
                        parar++;
                    }
                    if (image.at<uchar>(x+1,y+1)<127){
                        grow.at<uchar>(x+1,y+1)=255;
                        parar++;
                    }
                }
            }
        }
        imshow("Grow_Region", grow);
        waitKey(30);
    }
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Grow_Region.jpg", grow);
    waitKey();
}