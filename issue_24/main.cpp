#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

void on_mouse( int e, int x, int y, int d, void *ptr ){
    Point*p = (Point*)ptr;
    p->x = x;
    p->y = y;
}


int main(){
    int x,y;
    int val_b,val_r,val_g;
    
    Point p;
    
    Mat_<Vec3b> image = imread("/Users/wellcome/Desktop/PDI_LISTA/imagem.jpg",1);
    Mat grow(image.rows,image.cols,CV_8UC1);
    
    for(y=0;y<grow.rows;y++){
        for(x=0;x<grow.cols;x++)
            grow.at<uchar>(y,x)=0;
    }
    
    imshow("Aplicar_Semente",image);
    setMouseCallback("Aplicar_Semente",on_mouse, &p );
    waitKey(0);
    x= p.x ;
    y= p.y ;
    grow.at<uchar>(y,x)=255;
    val_b=image(y,x)[0];
    val_g=image(y,x)[1];
    val_r=image(y,x)[2];
    int Parar=0;
    int Continuar=1;
    while(Continuar!=Parar){
        Continuar=Parar;
        Parar=0;
        for(y=0;y<grow.rows;y++){
            for(x=0;x<grow.cols;x++){
                if(grow.at<uchar>(y,x)==255){
                    val_b=image(y,x)[0];
                    val_g=image(y,x)[1];
                    val_r=image(y,x)[2];
                    
                    if(val_b < 60 && val_g > 50 && val_r > 150 && grow.at<uchar>(y-1,x-1) != 255){
                        grow.at<uchar>(y-1,x-1)=255;
                        Parar++;
                    }
                    if(val_b < 60 && val_g > 50 && val_r > 150 && grow.at<uchar>(y-1,x)!= 255){
                        grow.at<uchar>(y-1,x)=255;
                        Parar++;
                    }
                    if(val_b < 60 && val_g > 50 && val_r > 150 && grow.at<uchar>(y-1,x+1) != 255){
                        grow.at<uchar>(y-1,x+1)=255;
                        Parar++;
                    }
                    if(val_b < 60 && val_g > 50 && val_r > 150 && grow.at<uchar>(y,x-1)!= 255){
                        grow.at<uchar>(y,x-1)=255;
                        Parar++;
                    }
                    if(val_b < 60 && val_g > 50 && val_r > 150 && grow.at<uchar>(y,x+1)!= 255){
                        grow.at<uchar>(y,x+1)=255;
                        Parar++;
                    }
                    if(val_b < 60 && val_g > 50 && val_r > 150 && grow.at<uchar>(y+1,x-1)!= 255){
                        grow.at<uchar>(y+1,x-1)=255;
                        Parar++;
                    }
                    if(val_b < 60 && val_g > 50 && val_r > 150 && grow.at<uchar>(y+1,x)!= 255){
                        grow.at<uchar>(y+1,x)=255;
                        Parar++;
                    }
                    if(val_b < 60 && val_g > 50 && val_r > 150 && grow.at<uchar>(y+1,x+1)!= 255){
                        grow.at<uchar>(y+1,x+1)=255;
                        Parar++;
                    }
                }
            }
        }
        imshow("Grow_region",grow);
        waitKey(30);
    }
    
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Grow_region.jpg",grow);
    waitKey(0);
}