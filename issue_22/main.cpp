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
Mat Centroide(Mat grow){
    int x,y;
    int max,mex,may,mey;
    int centrox=0,centroy=0;
    int cont=0;
    for (x=0; x<grow.rows; x++) {
        for (y=0; y<grow.cols; y++) {
            if (grow.at<uchar>(x,y)==255){
                if (cont==0){
                    max=x;mex=x;may=y;mey=y;cont++;
                }
                if(x>max)max=x;
                if(x<mex)mex=x;
                if(y>may)may=y;
                if(y<mey)mey=y;
            }
        }
    }
    centrox=((max-mex)/2+mex);
    centroy=((may-mey)/2+mey);
    
    Mat_<Vec3b> centroide(grow.rows,grow.cols,CV_8UC3);
    for (x=0; x<grow.rows; x++) {
        for (y=0; y<grow.cols; y++) {
            if (grow.at<uchar>(x,y)==255){
                centroide.at<Vec3b>(x,y)[0]=255;
                centroide.at<Vec3b>(x,y)[1]=0;
                centroide.at<Vec3b>(x,y)[2]=0;
            }
            else{
                centroide.at<Vec3b>(x,y)[0]=255;
                centroide.at<Vec3b>(x,y)[1]=255;
                centroide.at<Vec3b>(x,y)[2]=255;
            }
        }
    }
    circle(centroide, Point(centroy,centrox),3, CV_RGB(0,255,0),5,2,0);
    imshow("Centroide", centroide);
    return centroide;
}
int main(){
    Mat image=imread("/Users/wellcome/Desktop/PDI_LISTA/figura.jpg",0);
    Mat grow= grow_region(image);
    Mat centroide= Centroide(grow);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Grow_Region.jpg", grow);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Centroide.jpg", centroide);
    waitKey();
}