#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

void setPoint(int e,int x,int y,int d, void*ptr){
    Point*p= (Point*)ptr;
    p->x = x;
    p->y = y;
}

Mat grow_region(Mat image){
    int x,y;
    Point p;
    Mat grow(image.rows,image.cols,CV_8UC1);
    
    for (x=0; x<grow.rows; x++) {
        for (y=0; y<grow.cols; y++)
                grow.at<uchar>(x,y)=0;
    }
    imshow("Aplicar_Semente", image);
    setMouseCallback("Aplicar_Semente", setPoint, &p );
    waitKey(0);
    x= p.x;
    y= p.y;
    grow.at<uchar>(x,y)=255;
    
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
    
    return grow;
}

int main(){
    Mat image=imread("/Users/wellcome/Desktop/PDI_LISTA/img.png",0);
    Mat grow= grow_region(image);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Grow_Region.jpg", grow);
    waitKey();
}