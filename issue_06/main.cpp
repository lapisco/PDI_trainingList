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

int main(){
    Mat gray,canny;
    Mat image= imread("/Users/wellcome/Desktop/PDI_LISTA/arara.jpg”,1);
    imshow("Imagem", image);
    
    cvtColor(image, gray, CV_RGB2GRAY);
    imshow("Imagem Cinza", gray);
    
    Canny(gray, canny, 170, 200,3);
    imshow("Imagem Canny", canny);
    waitKey();
}