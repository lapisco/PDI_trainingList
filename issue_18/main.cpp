#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

Mat filtro_laplaciano_equalizado(Mat imagem)
{
    int x,y;
    int i,j;
    int soma;
    int ma,me;
    //Aplicacao do filtro laplaciano//
    Mat_<uchar> filtro(imagem.rows,imagem.cols,1);
    for(x=0;x<imagem.rows;x++)
    {
        for(y=0;y<imagem.cols;y++)
        {
            filtro.at<uchar>(x,y)=0;
        }
    }
    ma=0;
    me=0;
    for(y=1;y<imagem.rows-1;y++){
        for(x=1;x<imagem.cols-1;x++){
            soma=0;
            for(j=-1;j<=1;j++){
                for(i=-1;i<=1;i++){
                    if((x==x+i)&&(y==y+j)){
                        soma=(-8)*imagem.at<uchar>(y+j,x+i)+soma;
                    }
                    else{
                        soma= (1)*imagem.at<uchar>(y+j,x+i)+soma;
                    }
                }
            }
            soma=abs(soma);
            if(soma > ma)
                ma = soma;
            if(soma < me)
                me = soma;
            if(soma>255)
                filtro.at<uchar>(y,x)=255;
            else
                filtro.at<uchar>(y,x)=soma;
        }
    }
    imshow("Filtro_Laplace",filtro);
    //Aplicacao da equalizacao//
    for(y=1;y<imagem.rows-1;y++){
        for(x=1;x<imagem.cols-1;x++){
            if(filtro.at<uchar>(y,x)== me)
                filtro.at<uchar>(y,x)= 0;
            
            if(filtro.at<uchar>(y,x)== ma)
                filtro.at<uchar>(y,x) = 255;
            
            if((filtro.at<uchar>(y,x) > me) && (filtro.at<uchar>(y,x)< ma))
                filtro.at<uchar>(y,x)= (255*(filtro.at<uchar>(y,x)-me)/(ma-me));
        }
    }
    return filtro;
}

int main(){
    Mat image=imread("/Users/wellcome/Desktop/PDI_LISTA/arara.jpg",1);
    cvtColor(image, image, CV_RGB2GRAY);
    imshow("Imagem", image);
    Mat laplacian;
    
    laplacian=filtro_laplaciano_equalizado(image);
    imshow("Filtro_Laplace_Equalized", laplacian);
    
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/image_gray.jpg", image);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/image_laplacian_equalized.jpg", laplacian);
    waitKey();
}