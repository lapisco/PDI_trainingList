#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void detectar(Mat descriptors,Mat img_scene,Mat colorido)
{
    int i;
    float max_dist = 0;
    float min_dist = 100;
    Mat descriptors_scene,color=colorido;
    FlannBasedMatcher matcher;
    SiftFeatureDetector detector(20);//25-40
    SiftDescriptorExtractor extractor;
    Point2f soma;
    soma.x = 0;
    soma.y = 0;
    vector<Point2f> scene;
    vector<KeyPoint> keypoints_scene;
    vector< DMatch > matches;
    vector< DMatch > good_matches;
    
    detector.detect(img_scene,keypoints_scene);
    extractor.compute(img_scene,keypoints_scene, descriptors_scene);
    matcher.match(descriptors, descriptors_scene, matches);
    
    for(i=0;i<descriptors.rows; i++)
    {
        float dist = matches[i].distance;
        if( dist < min_dist )
        {
            min_dist = dist;
        }
        else if( dist > max_dist )
        {
            max_dist = dist;
        }
    }
    for(i=0;i<descriptors.rows;i++)
    {
        if( matches[i].distance<3*min_dist)
        {
            good_matches.push_back( matches[i]);
        }
    }
    
    for(i=0;i<good_matches.size();i++)
    {
        scene.push_back ( keypoints_scene[ good_matches[i].trainIdx ].pt );
        soma=soma+scene.at(i);
    }
    
    circle(color,Point( (int)((soma.x)/((float)scene.size())) , (int)((soma.y)/((float)scene.size())) ),45,CV_RGB(0,255,0),1,2,0);
    //resize(color,color,Size(512,384));
    imshow( "Object detection",color);
}
int main()
{
    Mat color,descriptors,frame;
    Mat img_1 = imread("/Users/iMacPedrosa/Desktop/img1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    resize(img_1,img_1,Size(256,192));
    SiftFeatureDetector detector(10);
    SiftDescriptorExtractor extractor;
    vector<KeyPoint> keypoints;
    detector.detect(img_1,keypoints);
    extractor.compute(img_1,keypoints,descriptors);
    
    VideoCapture cap(1);
    while(1){
        cap>>frame;
        resize(frame,color,Size(256,192));
        cvtColor(color,frame,CV_RGB2GRAY);
        detectar(descriptors,frame,color);
        if(waitKey(27) == 27){break;}
    }
    return 0;
}

//int main(){
//    
//    Mat img_1 = imread("/Users/iMacPedrosa/Desktop/img1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//    Mat img_2;
//    VideoCapture cap(0);
//    while (1) {
//        cap>>img_2;
//        cvtColor(img_2, img_2, CV_RGB2GRAY);
//        //imshow("Video capture", img_2);
//        Mat img_keypoints_1;
//        Mat img_keypoints_2;
//        
//        SiftFeatureDetector detector(100);
//        
//        vector<KeyPoint> keypoints_1;
//        vector<KeyPoint> keypoints_2;
//        
//        detector.detect(img_1,keypoints_1);
//        detector.detect(img_2,keypoints_2);
//        
//        
//        SiftDescriptorExtractor extractor;
//        
//        Mat descriptors_1, descriptors_2;
//        
//        extractor.compute( img_1, keypoints_1, descriptors_1 );
//        extractor.compute( img_2, keypoints_2, descriptors_2 );
//        
//        BFMatcher matcher(NORM_L2);
//        vector< DMatch > matches;
//        matcher.match( descriptors_1, descriptors_2, matches );
//        
////        Mat img_matches;
////        drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
//        
////        imshow("Matches", img_matches );
//        printf("1\n");
//    }
//}
