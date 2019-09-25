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


Mat image;
Mat redimens1;
Mat redimens2;

int main()
{
    image=imread("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/paisagem.jpg",1);
    namedWindow("Original Image",CV_WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    
    resize(image, redimens1, Size(160,120));
    namedWindow("Redimensionada 01",CV_WINDOW_AUTOSIZE);
    imshow("Redimensionada 01", redimens1);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/redimens1.jpg", redimens1);
    
    resize(image, redimens2, Size(640,480));
    namedWindow("Redimensionada 01",CV_WINDOW_AUTOSIZE);
    imshow("Redimensionada 01", redimens2);
    imwrite("/Users/wellcome/Dropbox/WELL_LAB/OpenCvLista/Results/redimens2.jpg", redimens2);
    
    waitKey(0);
    
}
