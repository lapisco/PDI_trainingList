#include <stdio.h>
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\nonfree\nonfree.hpp"
#include "opencv2\nonfree\features2d.hpp"

using namespace cv;
using namespace std;

int main(){
    
    Mat img_1 = imread("img1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_2 = imread("img2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    
    Mat img_keypoints_1;
    Mat img_keypoints_2;
    
    SurfFeatureDetector detector;
    
    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    
    detector.detect(img_1,keypoints_1);
    detector.detect(img_2,keypoints_2);
    
    
    SurfDescriptorExtractor extractor;
    
    Mat descriptors_1, descriptors_2;
    
    extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );
    
    BFMatcher matcher(NORM_L2);
    vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );
    
    Mat img_matches;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
    
    imshow("Matches", img_matches );
    
    waitKey(0);
    return 0;
}
