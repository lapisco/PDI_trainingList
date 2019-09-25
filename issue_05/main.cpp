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
Mat image_gray;
Mat image_median;
Mat image_blur;

int main()
{
    image_rgb= imread("/Users/iMacPedrosa/Desktop/WELL_PDI/arara.jpg",1);
    namedWindow("Image RGB",CV_WINDOW_AUTOSIZE);
    imshow("Image RGB", image_rgb);
    
    cvtColor(image_rgb, image_gray, CV_RGB2GRAY);
    namedWindow("Image Gray",CV_WINDOW_AUTOSIZE);
    imshow("Image Gray", image_gray);
    
    medianBlur(image_gray, image_median, CV_MEDIAN);
    namedWindow("Image Median",CV_WINDOW_AUTOSIZE);
    imshow("Image Median", image_median);
    imwrite("/Users/iMacPedrosa/Desktop/WELL_PDI/arara_median.jpg", image_median);
    
    blur(image_gray, image_blur, Size(3,3),Point(-1,-1));
    namedWindow("Image Blur",CV_WINDOW_AUTOSIZE);
    imshow("Image Blur", image_blur);
    imwrite("/Users/iMacPedrosa/Desktop/WELL_PDI/arara_blur.jpg", image_blur);
    
    waitKey(0);
    
}


