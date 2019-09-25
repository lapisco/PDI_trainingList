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

FILE*arquivo;

int main(){
    
    arquivo=fopen("/Users/wellcome/Desktop/PDI_LISTA/image.txt","r");
    char val[5000];
    int  val2 , linhas =-1 , colunas=-1;
    int x,y;
    //Pegar quantidade de Linhas//
    while(!feof(arquivo)){
        fgets(val,5000,arquivo);
        linhas++;
    }
    fclose(arquivo);
    //Pegar quantidade de Colunas//
    arquivo=fopen("/Users/wellcome/Desktop/PDI_LISTA/image.txt","r");
    while(!feof(arquivo)){
        fscanf(arquivo,"%d",&val2);
        colunas++;
    }
    colunas=colunas/linhas;
    fclose(arquivo);
    //Pegar do arquivo e criar imagem//
    arquivo=fopen("/Users/wellcome/Desktop/PDI_LISTA/image.txt","r");
    Mat_<uchar> img(linhas,colunas,1);
    for(y=0;y<linhas;y++)
    {
        for(x=0;x<colunas;x++)
        {
            fscanf(arquivo,"%d",&val2);
            img.at<uchar>(y,x)=val2;
        }
    }
    imshow("Imagem_Lida",img);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/imageLida.jpg",img);
    fclose(arquivo);
    waitKey(0);
}
