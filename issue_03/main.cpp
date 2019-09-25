#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

Mat image_rgb;

int main()
{
    
    image_rgb = imread("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/arara.jpg",1);
    
    namedWindow("Original Image",CV_WINDOW_AUTOSIZE);
    imshow("Original Image", image_rgb);
    
    Mat channel[3];
    split(image_rgb, channel);

    namedWindow("Channel Red",1);
    imshow("Channel Red", channel[0]);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Channel_Red.jpg", channel[0]);
    
    namedWindow("Channel Green",1);
    imshow("Channel Green", channel[1]);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Channel_Green.jpg", channel[1]);
    
    namedWindow("Channel Blue",1);
    imshow("Channel Blue", channel[2]);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/Channel_Blue.jpg", channel[2]);
    
    waitKey(0);
}