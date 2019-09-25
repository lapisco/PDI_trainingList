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
char address[100]="/Users/wellcome/Desktop/MomStat.txt";
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
    double Stat[10];
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
        Stat[0]=momentos.m00;
        Stat[1]=momentos.m10;
        Stat[2]=momentos.m01;
        Stat[3]=momentos.m20;
        Stat[4]=momentos.m11;
        Stat[5]=momentos.m02;
        Stat[6]=momentos.m30;
        Stat[7]=momentos.m21;
        Stat[8]=momentos.m12;
        Stat[9]=momentos.m03;
        for(int i=0; i<10; i++){
            
            if (i!=9){
                fprintf(file, "%f,", Stat[i]);
                printf("%f,", Stat[i]);
            }
            else{
                fprintf(file, "%f", Stat[i]);
                printf("%f", Stat[i]);
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