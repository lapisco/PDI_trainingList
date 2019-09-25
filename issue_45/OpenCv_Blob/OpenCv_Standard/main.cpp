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

CvRect rec;
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
    int cont=0;
    for(CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it){
        cont++;
        cvRenderBlob(imgLabel, (*it).second, thresholded, out, CV_BLOB_RENDER_COLOR);// Desenhar os contornos
        cvRenderBlob(imgLabel, (*it).second, thresholded, out, CV_BLOB_RENDER_BOUNDING_BOX);// Desenhar os Retangulos
        
        int area = (*it).second->area;
        int max = (*it).second->maxx;
        int mex = (*it).second->minx;
        int may = (*it).second->maxy;
        int mey = (*it).second->miny;
        printf("\nCaracteristicas do blob %d\n",cont);
        printf("ALTURA  : %d pixels\n",may-mey);
        printf("LARGURA : %d pixels\n",max-mex);
        printf("AREA    : %d pixels\n",area);
        
        rec = cvRect((*it).second->minx,(*it).second ->miny,((*it).second -> maxx - (*it).second -> minx),(*it).second -> maxy - (*it).second -> miny); //Subimagens
        cvSetImageROI(out,rec);
        char name[100];
        sprintf(name, "Janela_%d",cont);
        cvShowImage(name,out);
        cvWaitKey(0);
        cvResetImageROI(out);
    }
    unsigned long tam = blobs.size();
    printf("\nForam identificados %lu contornos\n",tam);
    cvShowImage("Original Image", image);
    cvShowImage("Image Blobs", out);
    cvSaveImage("/Users/wellcome/Desktop/OpenCv_Blob/Imagem_WithRectanglesDetected.jpg", out);
    cvWaitKey();
    
    return 0;
}
//***********OtherWayToMadeAContours*************//
//CvContourPolygon *polygon = cvConvertChainCodesToPolygon(&(*it).second->contour);
//CvContourPolygon *sPolygon = cvSimplifyPolygon(polygon, 0.1);
//
//cvRenderContourPolygon(sPolygon, output_blob, CV_RGB(0,0, 255));
//
//delete sPolygon;
//delete polygon;
