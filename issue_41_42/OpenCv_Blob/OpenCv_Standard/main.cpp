#include <iostream>
#include <iomanip>
#if (defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__) || (defined(__APPLE__) & defined(__MACH__)))
#include <opencv/cv.h>
#include <opencv/highgui.h>
#else
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif
#include <cvblob.h>
using namespace cvb;

int main(){
    IplImage *image = cvLoadImage("/Users/wellcome/Desktop/OpenCv_Blob/test.png", 1);
    IplImage *gray = cvCreateImage(cvGetSize(image),8,1);
    IplImage *thresholded = cvCreateImage(cvGetSize(image), 8, 1);
    cvCvtColor(image,gray,CV_RGB2GRAY);
    cvThreshold(gray,thresholded,127,255,CV_THRESH_BINARY);
    
    IplImage *imgLabel = cvCreateImage(cvGetSize(image),IPL_DEPTH_LABEL,1);
    CvBlobs blobs;
    
    unsigned int result = cvLabel(thresholded, imgLabel, blobs);
    cvFilterByArea(blobs,1,100000);
    
    unsigned long tam = blobs.size();
    printf("\nForam identificados %lu contornos\n",tam);
    cvShowImage("Original Image", image);
    cvWaitKey();
    return 0;
}
