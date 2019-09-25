#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

FILE*file;
FILE*tempos;
char address[100]="/Users/wellcome/Desktop/MomCent.txt";
char addressTime[100]="/Users/wellcome/Desktop/Time.txt";
void abrirFile(){
    file=fopen(address, "w");
    fclose(file);
}
void abrirFileTime(){
    file=fopen(address, "w");
    fclose(file);
}
int main(){
    clock_t Tmp[2];
    abrirFile();
    abrirFileTime();
    Mat image;
    double Central[7];
    int x=0;
    while (x!=200) {
        Tmp[0]=clock();
        x++;
        
        char add[200];
        sprintf(add, "/Users/wellcome/Desktop/BancoDeImagens/s (%d).png",x);
        image= imread(add,IMREAD_COLOR);
        cvtColor(image, image, CV_RGB2GRAY);
        
        file=fopen(address, "a");
        Moments momentos =  moments(image,true);
        Central[0]=momentos.mu20;
        Central[1]=momentos.mu11;
        Central[2]=momentos.mu02;
        Central[3]=momentos.mu30;
        Central[4]=momentos.mu21;
        Central[5]=momentos.mu12;
        Central[6]=momentos.mu03;
        
        for(int i=0; i<7; i++){
            
            if (i!=6){
                fprintf(file, "%f,", Central[i]);
                printf("%f,", Central[i]);
            }
            else{
                fprintf(file, "%f", Central[i]);
                printf("%f", Central[i]);
            }
        }
        fprintf(file, "\n");
        printf("\n");
        fclose(file);
        Tmp[1]=clock();
        double Time= (Tmp[1]-Tmp[0])*1000.0/CLOCKS_PER_SEC;//Em MiliSegundos
        tempos=fopen(addressTime, "a");
        fprintf(tempos, "%f\n",Time);
        fclose(tempos);
    }
}