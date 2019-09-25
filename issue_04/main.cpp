#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

Mat img_rgb;
Mat img_hsv;

int main()
{
    
    img_rgb = imread("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/arara.jpg",1);
    namedWindow("Image RGB",CV_WINDOW_AUTOSIZE);
    imshow("Image RGB", img_rgb);
    
    cvtColor(img_rgb, img_hsv, CV_RGB2HSV);
    namedWindow("Image HSV",CV_WINDOW_AUTOSIZE);//H-Hue(Tonalidade),S-Saturation(Saturação),V-Value(Brilho)
    imshow("Image HSV", img_hsv);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Image_HSV.jpg", img_hsv);
    
    namedWindow("Channel H",CV_WINDOW_AUTOSIZE);
    namedWindow("Channel S",CV_WINDOW_AUTOSIZE);
    namedWindow("Channel V",CV_WINDOW_AUTOSIZE);
    
    Mat channel[3];
    split(img_hsv, channel);
    
    imshow("Channel H", channel[0]);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Channel_H(Tonalidade).jpg", channel[0]);
    imshow("Channel S", channel[1]);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Channel_S(Saturação).jpg", channel[1]);
    imshow("Channel V", channel[2]);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Channel_V(Brilho).jpg", channel[2]);
    
    waitKey(0);
}
