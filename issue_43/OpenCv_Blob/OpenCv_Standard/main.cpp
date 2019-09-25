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
    IplImage *out= cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 3);
    unsigned int result = cvLabel(thresholded, imgLabel, blobs);
    cvFilterByArea(blobs,1,100000);
    cvZero(out);
    
    for(CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it){
        cvRenderBlob(imgLabel, (*it).second, thresholded, out, CV_BLOB_RENDER_COLOR);// Desenhar os contornos
        cvRenderBlob(imgLabel, (*it).second, thresholded, out, CV_BLOB_RENDER_BOUNDING_BOX);// Desenhar os Retangulos
    }
    unsigned long tam = blobs.size();
    printf("\nForam identificados %lu contornos\n",tam);
    cvShowImage("Original Image", image);
    cvShowImage("Image Blobs", out);
    cvSaveImage("/Users/wellcome/Desktop/OpenCv_Blob/Imagem_WithRectanglesDetected.jpg", out);
    cvWaitKey();
    
    return 0;
}
