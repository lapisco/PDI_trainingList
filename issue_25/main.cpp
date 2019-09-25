#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

void SetPoint( int e, int x, int y, int d, void *ptr){
    Point*p = (Point*)ptr;
    p->x = x;
    p->y = y;
}
int Channel_Selection(int channel,int contador){
    if (contador==0){
        channel=2;
        return channel;
    }
    if (contador==1){
        channel=0;
        return channel;
    }
    if (contador==2){
        channel=1;
        return channel;
    }
    return 0;
}
Mat grow_region(Mat imagem)
{
    int x,y;
    Mat_<Vec3b> grow(imagem.rows,imagem.cols,CV_8UC3);
    for(y=0;y<grow.rows;y++){
        for(x=0;x<grow.cols;x++){
            grow(y,x)[0]=0;
            grow(y,x)[1]=0;
            grow(y,x)[2]=0;
        }
    }
    int canal;
    int cont=0;
    while(cont<3)
    {
        canal=Channel_Selection(canal, cont);
        Point p;
        
        imshow("Aplicar_Semente",imagem);
        setMouseCallback("Aplicar_Semente",SetPoint, &p );
        waitKey(0);
        x= p.x ;
        y= p.y ;
        grow(y,x)[canal]=255;
        int Continuar=1;
        int Parar=0;
        while(Continuar!=Parar)
        {
            Continuar=Parar;
            Parar=0;
            for(y=0;y<imagem.rows;y++)
            {
                for(x=0;x<imagem.cols;x++)
                {
                    if(grow(y,x)[canal]==255)
                    {
                        if(imagem.at<uchar>(y+1,x-1)<127)
                        {
                            grow(y+1,x-1)[canal]=255;
                            Parar++;
                        }
                        if(imagem.at<uchar>(y,x-1)<127)
                        {
                            grow(y,x-1)[canal]=255;
                            Parar++;
                        }
                        if(imagem.at<uchar>(y-1,x-1)<127)
                        {
                            grow(y-1,x-1)[canal]=255;
                            Parar++;
                        }
                        if(imagem.at<uchar>(y+1,x)<127)
                        {
                            grow(y+1,x)[canal]=255;
                            Parar++;
                        }
                        if(imagem.at<uchar>(y+1,x+1)<127)
                        {
                            grow(y+1,x+1)[canal]=255;
                            Parar++;
                        }
                        if(imagem.at<uchar>(y,x+1)<127)
                        {
                            grow(y,x+1)[canal]=255;
                            Parar++;
                        }
                        if(imagem.at<uchar>(y-1,x+1)<127)
                        {
                            grow(y-1,x+1)[canal]=255;
                            Parar++;
                        }
                        if(imagem.at<uchar>(y-1,x)<127)
                        {
                            grow(y-1,x)[canal]=255;
                            Parar++;
                        }
                    }
                }
            }
            imshow("Grow_region",grow);
            waitKey(30);
        }
        cont++;
    }
    return grow;
}

int main(){
    Mat img_rgb = imread("/Users/wellcome/Desktop/PDI_LISTA/img.png",1);
    Mat gray;
    cvtColor(img_rgb, gray, CV_RGB2GRAY);
    Mat grow=grow_region(gray);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Grow_region.jpg",grow);
    waitKey(0);
}
