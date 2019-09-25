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

int main()
{
    Mat_<uchar> cinza;
    Mat color = imread("/Users/wellcome/Desktop/PDI_LISTA/arara.jpg",1);
    cvtColor(color,cinza,CV_RGB2GRAY);
    
    int Gx,Gy;
    Mat_<uchar> filtro_sobel(cinza.rows,cinza.cols,1);
    
    for(int x=0; x<filtro_sobel.rows; x++){
        for(int y=0; y<filtro_sobel.cols; y++){
            filtro_sobel.at<uchar>(x,y)=0;
        }
    }
    
    for(int y=1; y< filtro_sobel.rows -1 ; y++)
    {
        for(int x=1;x< filtro_sobel.cols -1 ;x++)
        {
            //Aplicacao do Sobel, Varrendo verticalmente e horizontalmente com as mascaras sobel.
            //   Horizontal     Vertical
            // | +1 +2 +1 |    | -1 0 +1 |
            // |  0  0  0 |    | -2 0 +2 |
            // | -1 -2 -1 |    | -1 0 +1 |
            
            Gx = (1)*cinza.at<uchar>(y-1,x-1) + (2)*cinza.at<uchar>(y-1,x) + (1)*cinza.at<uchar>(y-1,x+1) + (-1)*cinza.at<uchar>(y+1,x-1) + (-2)*cinza.at<uchar>(y+1,x) + (-1)*cinza.at<uchar>(y+1,x+1);
            Gy = (1)*cinza.at<uchar>(y-1,x-1) + (-1)*cinza.at<uchar>(y-1,x+1) + (2)*cinza.at<uchar>(y,x-1) + (-2)*cinza.at<uchar>(y,x+1) + (1)*cinza.at<uchar>(y+1,x-1) + (-1)*cinza.at<uchar>(y+1,x+1);
            //    ____________
            // G=\|Gx^2 + Gy^2 '
            //
            filtro_sobel.at<uchar>(y,x) = (int)sqrt(pow(Gx,2)+pow(Gy,2));
        }
    }
    imshow("Filtro_sobel",filtro_sobel);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/arara_sobel.jpg", filtro_sobel);
    waitKey(0);
}