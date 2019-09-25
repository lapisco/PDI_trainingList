#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
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
    Mat_<Vec3b> colorida(imagem.rows,imagem.cols,CV_8UC3);
    
    for(y=0;y<grow.rows;y++){
        for(x=0;x<grow.cols;x++){
            grow.at<uchar>(y,x)=0;
            colorida(y,x)[0]=0;
            colorida(y,x)[1]=0;
            colorida(y,x)[2]=0;
        }
    }
    
    for(y2=0;y2<imagem.rows;y2++){
        for(x2=0;x2<imagem.cols;x2++){
            if((imagem.at<uchar>(y2,x2)==0)&&(grow.at<uchar>(y2,x2)==0)){
                grow.at<uchar>(y2,x2)=255;
                cores(val_b,val_g,val_r,cont);
                cont++;
                int atual=0,anterior=1;
                
                while(atual!=anterior){
                    anterior=atual;
                    atual=0;
                    
                    for(y=0;y<grow.rows;y++){
                        for(x=0;x<grow.cols;x++){
                            if(grow.at<uchar>(y,x)==255){
                                if((imagem.at<uchar>(y+1,x-1)<127)&&(grow.at<uchar>(y+1,x-1)!=255)){
                                    grow.at<uchar>(y+1,x-1)=255;
                                    colorida(y+1,x-1)[0]= val_b;
                                    colorida(y+1,x-1)[1]= val_g;
                                    colorida(y+1,x-1)[2]= val_r;
                                    atual++;
                                }
                                if((imagem.at<uchar>(y,x-1)<127)&&(grow.at<uchar>(y,x-1)!=255)){
                                    grow.at<uchar>(y,x-1)=255;
                                    colorida(y,x-1)[0]=val_b;
                                    colorida(y,x-1)[1]=val_g;
                                    colorida(y,x-1)[2]=val_r;
                                    atual++;
                                }
                                if((imagem.at<uchar>(y-1,x-1)<127)&&(grow.at<uchar>(y-1,x-1)!=255)){
                                    grow.at<uchar>(y-1,x-1)=255;
                                    colorida(y-1,x-1)[0]=val_b;
                                    colorida(y-1,x-1)[1]=val_g;
                                    colorida(y-1,x-1)[2]=val_r;
                                    atual++;
                                }
                                if((imagem.at<uchar>(y+1,x)<127)&&(grow.at<uchar>(y+1,x)!=255)){
                                    grow.at<uchar>(y+1,x)=255;
                                    colorida(y+1,x)[0]=val_b;
                                    colorida(y+1,x)[1]=val_g;
                                    colorida(y+1,x)[2]=val_r;
                                    atual++;
                                }
                                if((imagem.at<uchar>(y+1,x+1)<127)&&(grow.at<uchar>(y+1,x+1)!=255)){
                                    grow.at<uchar>(y+1,x+1)=255;
                                    colorida(y+1,x+1)[0]=val_b;
                                    colorida(y+1,x+1)[1]=val_g;
                                    colorida(y+1,x+1)[2]=val_r;
                                    atual++;
                                }
                                if((imagem.at<uchar>(y,x+1)<127)&&(grow.at<uchar>(y,x+1)!=255)){
                                    grow.at<uchar>(y,x+1)=255;
                                    colorida(y,x+1)[0]=val_b;
                                    colorida(y,x+1)[1]=val_g;
                                    colorida(y,x+1)[2]=val_r;
                                    atual++;
                                }
                                if((imagem.at<uchar>(y-1,x+1)<127)&&(grow.at<uchar>(y-1,x+1)!=255)){
                                    grow.at<uchar>(y-1,x+1)=255;
                                    colorida(y-1,x+1)[0]=val_b;
                                    colorida(y-1,x+1)[1]=val_g;
                                    colorida(y-1,x+1)[2]=val_r;
                                    atual++;
                                }
                                if((imagem.at<uchar>(y-1,x)<127)&&(grow.at<uchar>(y-1,x)!=255)){
                                    grow.at<uchar>(y-1,x)=255;
                                    colorida(y-1,x)[0]=val_b;
                                    colorida(y-1,x)[1]=val_g;
                                    colorida(y-1,x)[2]=val_r;
                                    atual++;
                                }
                            }
                        }
                    }
                    imshow("Grow_region",colorida);
                    waitKey(30);
                }
            }
        }
    }
    printf("\nForam detectados %d objetos",cont);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Grow_Region.jpg", grow);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/Grow_Region_color.jpg", colorida);
    return grow;
}
    
int main(){
    Mat img = imread("/Users/wellcome/Desktop/PDI_LISTA/img.png",0);
    Mat grow=grow_region(img);
    waitKey(0);
}