#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

Mat histograma(Mat imagem){
    int vet[256],x,y,num;
    int ma=0;
    for (x=0; x<256; x++) {
        vet[x]=0;
    }
    for (x=0; x<imagem.rows; x++) {
        for (y=0; y<imagem.cols; y++) {
            num=0;
            num=imagem.at<uchar>(x,y);
            vet[num]++;
        }
    }
    for (x=0; x<256; x++) {
        if (ma==0)
            ma=vet[x];
        if (vet[x]>ma)
            ma=vet[x];
    }
    int max=ma/200;
    for (x=0; x<256; x++) {
        vet[x]=vet[x]/max;
    }
    Mat_<Vec3b> histogram(200,256,CV_8UC3);
    for (x=0; x<histogram.rows; x++) {
        for (y=0; y<histogram.cols; y++) {
            if (x>=histogram.rows-vet[y]) {
                histogram(x,y)[0]=0;
                histogram(x,y)[1]=0;
                histogram(x,y)[2]=0;
            }
            else{
                histogram(x,y)[0]=255;
                histogram(x,y)[1]=255;
                histogram(x,y)[2]=255;
            }
        }
    }
    return histogram;
}

int main(){
    Mat sobel,hist,histsobel;
    Mat image=imread("/Users/wellcome/Desktop/PDI_LISTA/arara.jpg",1);
    cvtColor(image, image, CV_RGB2GRAY);
    hist=histograma(image);
    
    //Aplicação do Sobel//
    int Gx,Gy;
    Mat_<uchar> filtro_sobel(image.rows,image.cols,1);
    
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
            
            Gx = (1)*image.at<uchar>(y-1,x-1) + (2)*image.at<uchar>(y-1,x) + (1)*image.at<uchar>(y-1,x+1) + (-1)*image.at<uchar>(y+1,x-1) + (-2)*image.at<uchar>(y+1,x) + (-1)*image.at<uchar>(y+1,x+1);
            Gy = (1)*image.at<uchar>(y-1,x-1) + (-1)*image.at<uchar>(y-1,x+1) + (2)*image.at<uchar>(y,x-1) + (-2)*image.at<uchar>(y,x+1) + (1)*image.at<uchar>(y+1,x-1) + (-1)*image.at<uchar>(y+1,x+1);
            //    ____________
            // G=\|Gx^2 + Gy^2 '
            //
            filtro_sobel.at<uchar>(y,x) = (int)sqrt(pow(Gx,2)+pow(Gy,2));
        }
    }
    histsobel=histograma(filtro_sobel);
    
    imshow("Imagem", image);
    imshow("Histograma_Image", hist);
    imshow("Image_sobel", filtro_sobel);
    imshow("Histograma_sobel", histsobel);
    
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/image_gray.jpg", image);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/histograma_image.jpg", hist);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/image_sobel.jpg", filtro_sobel);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/histograma_sobel.jpg", histsobel);
    waitKey();
}