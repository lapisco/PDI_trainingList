#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

void cores(int &val_b,int &val_g,int &val_r,int cont){
    if(cont==0){
        val_r = 255;
        val_b = 0;
        val_g = 0;
    }
    else if(cont==1){
        val_r = 0;
        val_b = 255;
        val_g = 0;
    }
    else if(cont==2){
        val_r=0;
        val_b=0;
        val_g=255;
    }
    else if (cont%2!=0){
        val_r = val_r + 5*cont;
        val_b = val_b + 25*cont;
        val_g = val_g + 17*cont;
    }
    else if(cont%2==0){
        val_r = val_r+ 8;
        val_b = val_b+ 17*cont;
        val_g = val_g+ 25*(cont+1);
    }
    else if(val_r==255||val_b==255||val_g==255){
        val_r=8;
        val_b=9;
        val_g=10;
    }
}


Mat grow_region(Mat imagem){
    int x,y;
    int x2,y2;
    int cont=0;
    int val_b=255,val_g=0,val_r=0;
    
    Mat grow(imagem.rows,imagem.cols,CV_8UC1);
    Mat gray;
    cvtColor(imagem, gray, CV_RGB2GRAY);
    Mat_<Vec3b> colorida(imagem.rows,imagem.cols,CV_8UC3);
    
    for(x=0;x<grow.rows;x++){
        for(y=0;y<grow.cols;y++){
            grow.at<uchar>(x,y)=0;
            colorida(x,y)[0]=0;
            colorida(x,y)[1]=0;
            colorida(x,y)[2]=0;
        }
    }
    
    for(x2=0;x2<gray.rows;x2++){
        for(y2=0;y2<gray.cols;y2++){
            if((gray.at<uchar>(x2,y2)==0)&&(grow.at<uchar>(x2,y2)==0)){
                grow.at<uchar>(x2,y2)=255;
                cores(val_b,val_g,val_r,cont);
                int max=x2,may=y2,mex=x2,mey=y2;
                cont++;
                
                int atual=0,anterior=1;
                while(atual!=anterior){
                    anterior=atual;
                    atual=0;
                    
                    for(x=0;x<grow.rows;x++){
                        for(y=0;y<grow.cols;y++){
                            if(grow.at<uchar>(x,y)==255){
                                if((gray.at<uchar>(x+1,y-1)<127)&&(grow.at<uchar>(x+1,y-1)!=255)){
                                    grow.at<uchar>(x+1,y-1)=255;
                                    colorida(x+1,y-1)[0]= val_b;
                                    colorida(x+1,y-1)[1]= val_g;
                                    colorida(x+1,y-1)[2]= val_r;
                                    if (max<x+1)max=x+1;
                                    if (may<y-1)may=y-1;
                                    if (mex>x+1)mex=x+1;
                                    if (mey>y-1)mey=y-1;
                                    atual++;
                                }
                                if((gray.at<uchar>(x,y-1)<127)&&(grow.at<uchar>(x,y-1)!=255)){
                                    grow.at<uchar>(x,y-1)=255;
                                    colorida(x,y-1)[0]=val_b;
                                    colorida(x,y-1)[1]=val_g;
                                    colorida(x,y-1)[2]=val_r;
                                    if (max<x)max=x;
                                    if (may<y-1)may=y-1;
                                    if (mex>x)mex=x;
                                    if (mey>y-1)mey=y-1;
                                    atual++;
                                }
                                if((gray.at<uchar>(x-1,y-1)<127)&&(grow.at<uchar>(x-1,y-1)!=255)){
                                    grow.at<uchar>(x-1,y-1)=255;
                                    colorida(x-1,y-1)[0]=val_b;
                                    colorida(x-1,y-1)[1]=val_g;
                                    colorida(x-1,y-1)[2]=val_r;
                                    if (max<x-1)max=x-1;
                                    if (may<y-1)may=y-1;
                                    if (mex>x-1)mex=x-1;
                                    if (mey>y-1)mey=y-1;
                                    atual++;
                                }
                                if((gray.at<uchar>(x+1,y)<127)&&(grow.at<uchar>(x+1,y)!=255)){
                                    grow.at<uchar>(x+1,y)=255;
                                    colorida(x+1,y)[0]=val_b;
                                    colorida(x+1,y)[1]=val_g;
                                    colorida(x+1,y)[2]=val_r;
                                    if (max<x+1)max=x+1;
                                    if (may<y)may=y;
                                    if (mex>x+1)mex=x+1;
                                    if (mey>y)mey=y;
                                    atual++;
                                }
                                if((gray.at<uchar>(x+1,y+1)<127)&&(grow.at<uchar>(x+1,y+1)!=255)){
                                    grow.at<uchar>(x+1,y+1)=255;
                                    colorida(x+1,y+1)[0]=val_b;
                                    colorida(x+1,y+1)[1]=val_g;
                                    colorida(x+1,y+1)[2]=val_r;
                                    if (max<x+1)max=x+1;
                                    if (may<y+1)may=y+1;
                                    if (mex>x+1)mex=x+1;
                                    if (mey>y+1)mey=y+1;
                                    atual++;
                                }
                                if((gray.at<uchar>(x,y+1)<127)&&(grow.at<uchar>(x,y+1)!=255)){
                                    grow.at<uchar>(x,y+1)=255;
                                    colorida(x,y+1)[0]=val_b;
                                    colorida(x,y+1)[1]=val_g;
                                    colorida(x,y+1)[2]=val_r;
                                    if (max<x)max=x;
                                    if (may<y+1)may=y+1;
                                    if (mex>x)mex=x;
                                    if (mey>y+1)mey=y+1;
                                    atual++;
                                }
                                if((gray.at<uchar>(x-1,y+1)<127)&&(grow.at<uchar>(x-1,y+1)!=255)){
                                    grow.at<uchar>(x-1,y+1)=255;
                                    colorida(x-1,y+1)[0]=val_b;
                                    colorida(x-1,y+1)[1]=val_g;
                                    colorida(x-1,y+1)[2]=val_r;
                                    if (max<x-1)max=x-1;
                                    if (may<y+1)may=y+1;
                                    if (mex>x-1)mex=x-1;
                                    if (mey>y+1)mey=y+1;
                                    atual++;
                                }
                                if((gray.at<uchar>(x-1,y)<127)&&(grow.at<uchar>(x-1,y)!=255)){
                                    grow.at<uchar>(x-1,y)=255;
                                    colorida(x-1,y)[0]=val_b;
                                    colorida(x-1,y)[1]=val_g;
                                    colorida(x-1,y)[2]=val_r;
                                    if (max<x-1)max=x-1;
                                    if (may<y)may=y;
                                    if (mex>x-1)mex=x-1;
                                    if (mey>y)mey=y;
                                    atual++;
                                }
                            }
                        }
                    }
                    imshow("Grow_region",colorida);
                    waitKey(30);
                }
                
                Mat window(max-mex,may-mey,CV_8UC1);
                for (int a=mex; a<max; a++) {
                    for (int b=mey; b<may; b++) {
                        window.at<uchar>(a,b)=0;
                    }
                }
                int x3=0;
                for (int xx=mex; xx<max; xx++) {
                    int y3=0;
                    for (int yy=mey; yy<may; yy++) {
                        window.at<uchar>(x3,y3)=grow.at<uchar>(xx,yy);
                        y3++;
                    }
                    x3++;
                }
                char name[50];
                char endereco[200];
                sprintf(name, "Janela_%d",cont);
                sprintf(endereco,"/Users/wellcome/Desktop/Janela_%d.jpg",cont);
                imshow("Janela", window);
                imwrite(endereco, window);
                waitKey();
            }
        }
    }
    printf("\nForam detectados %d objetos",cont);
    imwrite("/Users/wellcome/Desktop/Grow_Region.jpg", grow);
    imwrite("/Users/wellcome/Desktop/Grow_Region_color.jpg", colorida);
    return grow;
}

int main(){
    Mat_<Vec3b> img = imread("/Users/wellcome/Desktop/img.png",1);
    Mat grow=grow_region(img);
    waitKey(0);
}