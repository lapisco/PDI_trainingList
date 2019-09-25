#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

FILE*arquivo;
Mat image;
Mat gray;
Mat limi;

int main()
{
    image=imread("/Users/wellcome/Desktop/PDI_LISTA/paisagem.jpg",1);
    namedWindow("Original Image",CV_WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    
    cvtColor(image, gray, CV_RGB2GRAY);
    namedWindow("Image Gray",CV_WINDOW_AUTOSIZE);
    imshow("Image Gray", gray);
    
    arquivo=fopen("/Users/wellcome/Desktop/PDI_LISTA/image.txt", "w");
    int x,y;
    Mat_<uchar> img(gray.rows,gray.cols,1);
    for(y = 0; y < gray.rows; y++){
        for(x = 0; x < gray.cols; x++)
        {
            if(gray.at<uchar>(y,x)<127)
            {
                img.at<uchar>(y,x)=0;
            }
            else
            {
                img.at<uchar>(y,x)=255;
            }
            fprintf(arquivo,"%d ",img.at<uchar>(y,x));
        }
        fprintf(arquivo,"\n");
    }
    imshow("Imagem_Limiarizada",img);
    imwrite("Imagem_Limiarizada.jpg",img);
    fclose(arquivo);
    waitKey(0);
    
}