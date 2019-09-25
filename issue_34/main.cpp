#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;

int main(){
    Mat image=imread("/Users/wellcome/Desktop/PDI_LISTA/img.png",1);
    cvtColor(image, image, CV_RGB2GRAY);
    double area;
    Canny(image, image, 50,100,3);
    threshold(image, image, 127, 255, CV_THRESH_BINARY);
    imshow("Trans_canny", image);
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/contours.jpg", image);
    
    vector<vector<Point> > contours;
    findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    printf("Existe %d contornos!\n",contours.size());
    
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    
    for(int i=0; i<contours.size(); i++ ){
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }
    
    Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
    int cont=0;
    char nome[100];
    char namesave[200];
    
    for (int i=0; i<contours.size();i++){
        drawContours(drawing, contours_poly, i, Scalar(0,255,0));
        rectangle(drawing, Point(boundRect[i].tl().x,boundRect[i].tl().y), Point(boundRect[i].br().x,boundRect[i].br().y), Scalar(0,255,0));
        area= contourArea(contours_poly[i]);
        printf("Area(%d)=%0.1f\n",i+1,area);
        
        Mat_<uchar> gray;
        cvtColor(drawing,gray,CV_RGB2GRAY);
        Mat_<uchar> janela(boundRect[i].br().y-boundRect[i].tl().y,boundRect[i].br().x-boundRect[i].tl().x,1);
        int x,y;
        int x2=0,y2=0;
        for(y=boundRect[i].tl().y;y< boundRect[i].br().y;y++){
            x2=0;
            for(x=boundRect[i].tl().x;x< boundRect[i].br().x;x++){
                janela.at<uchar>(y2,x2)=gray.at<uchar>(y,x);
                x2++;
            }
            y2++;
        }
        cont++;
        sprintf(nome,"Janela_%d.jpg",cont);
        imshow(nome,janela);
        
        sprintf(namesave,"/Users/wellcome/Desktop/PDI_LISTA/Janela_%d.jpg",cont);
        imwrite(namesave, janela);
    }
    imshow("Contours", drawing );
    imwrite("/Users/wellcome/Desktop/PDI_LISTA/drawing.jpg", drawing);
    waitKey(0);
}

////////////////////////////CvFindContours////////////////////////////////////////////////////
//Mode:
//    CV_RETR_EXTERNAL -  Recupera apenas os contornos exteriores extremas.
//    CV_RETR_LIST - Recupera todos os contornos , sem estabelecer quaisquer relacoes hierarquicas .
//    CV_RETR_CCOMP - Recupera todos os contornos e as organiza em uma hierarquia de dois niveis : no nivel superior sao os limites externos dos componentes, no segundo nivel sao os limites dos buracos.
//    CV_RETR_TREE - recupera todos os contornos e reconstroi a hierarquia completa de contornos aninhados .
//
//Metodo:
//    CV_CHAIN_APPROX_NONE - Absolutamente todos os pontos de contorno. Ou seja, a cada 2 pontos de um contorno armazenados com este metodo sao vizinhos 8- conectadas entre si.
//    CV_CHAIN_APPROX_SIMPLE - Comprime segmentos horizontais , verticais e diagonais e deixa apenas os seus pontos finais.
//    CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS - Aplica um dos sabores do algoritmo de aproximacao cadeia Teh- Chin , ver TehChin89 .