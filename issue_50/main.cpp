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
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

FILE* file;
FILE* tempo;

char address[200]="/Users/wellcome/Desktop/GLCM.txt";
char addressTime[200]="/Users/wellcome/Desktop/Time.txt";
void abrirFile(){
    file=fopen(address, "w");
    fclose(file);
}
void abrirFileTime(){
    tempo=fopen(address, "w");
    fclose(file);
}
void glcm(Mat &img){
    
    float energy=0,contrast=0,homogenity=0;
    float IDM=0,entropy=0, dissimilarity=0;
    float asm1=0, correlation=0;
    float mean1=0, mean2=0, omegai=0, omegaj=0;
    float variance=0,sumEntropy=0,sumVariance=0,sumAverage=0;
    float diferenceEntropy=0,diferenceVariance=0;
    
    int row=img.rows,col=img.cols;
    Mat gl=Mat::zeros(256,256,CV_32FC1);
    
    //creating glcm matrix with 256 levels,radius=1 and in the horizontal direction
    for(int i=0;i<row;i++)
        for(int j=0;j<col-1;j++)
            gl.at<float>(img.at<uchar>(i,j),img.at<uchar>(i,j+1))=gl.at<float>(img.at<uchar>(i,j),img.at<uchar>(i,j+1))+1;
    
    // normalizing glcm matrix for parameter determination
    gl=gl+gl.t();
    gl=gl/sum(gl)[0];
    
    
    for(int i=0;i<256;i++){
        for(int j=0;j<256;j++){
            contrast=contrast+(i-j)*(i-j)*gl.at<float>(i,j);
            homogenity=homogenity+gl.at<float>(i,j)/(1+abs(i-j)); // No denominador È ( i - j ) ao quadrado //
            dissimilarity=dissimilarity+gl.at<float>(i,j)*(abs(i-j));
            asm1=asm1+(gl.at<float>(i,j)*gl.at<float>(i,j));
            energy=energy+sqrt(asm1);
            
            if(i!=j){
                IDM=IDM+gl.at<float>(i,j)/((i-j)*(i-j)); //Taking k=2; // IDM no artigo o denominador È (1 + (i-j)^2)
            }
            if(gl.at<float>(i,j)!=0){
                entropy=entropy-gl.at<float>(i,j)*log10(gl.at<float>(i,j));
            }
            mean1=mean1+i*gl.at<float>(i,j);
            mean2=mean2+j*gl.at<float>(i,j);
            omegai=omegai+sqrt(gl.at<float>(i,j)*(i-mean1)*(i-mean2));
            omegaj=omegaj+sqrt(gl.at<float>(i,j)*(j-mean2)*(j-mean2));
            if (omegai!=0 && omegaj!=0){
                correlation=correlation+((((i*j)*(gl.at<float>(i,j)))-(mean1*mean2))/(omegai*omegaj));
            }
            variance+=((i-(mean1+mean2)/2)*(i-(mean1+mean2)/2))*gl.at<float>(i,j);
            for (int s=2; s<513; s++) {
                if (i+j==s && gl.at<float>(s)>0) {
                    sumEntropy+=(-1)*(gl.at<float>(s))*log10((gl.at<float>(s)));
                    sumVariance+=(i-sumEntropy)*(i-sumEntropy)*(gl.at<float>(s));
                    sumAverage+=i*(gl.at<float>(s));
                }
            }
            for (int s=0; s<256; s++) {
                if (i-j==s && gl.at<float>(s)>0) {
                    diferenceEntropy+=(-1)*(gl.at<float>(s))*log10(gl.at<float>(s));
                    diferenceVariance+=((i-(mean1+mean2)/2)*(i-(mean1+mean2)/2))*gl.at<float>(s);
                }
            }
        }
    }
    //Results
    cout<<"energy="<<energy<<endl;
    cout<<"contrast="<<contrast<<endl;
    cout<<"homogenity="<<homogenity<<endl;
    cout<<"IDM="<<IDM<<endl;
    cout<<"entropy="<<entropy<<endl;
    cout<<"dissimilarity="<<dissimilarity<<endl;
    cout<<"AngularSecondMoment="<<asm1<<endl;
    cout<<"correlation="<<correlation<<endl;
    cout<<"variance="<<variance<<endl;
    cout<<"sumEntropy="<<sumEntropy<<endl;
    cout<<"sumVariance="<<sumVariance<<endl;
    cout<<"sumAverage="<<sumAverage<<endl;
    cout<<"diferenceEntropy="<<diferenceEntropy<<endl;
    cout<<"diferenceVariance="<<diferenceVariance<<endl<<endl<<endl;

    //Para FILE
    file=fopen(address, "a");
    fprintf(file,"%f,",entropy);
    fprintf(file,"%f,",energy);
    fprintf(file,"%f,",homogenity);
    fprintf(file,"%f,",contrast);
    fprintf(file,"%f,",IDM);
    fprintf(file,"%f,",dissimilarity);
    fprintf(file,"%f,",asm1);
    fprintf(file,"%f,",correlation);
    fprintf(file,"%f,", variance);
    fprintf(file,"%f,", sumEntropy);
    fprintf(file,"%f,", sumVariance);
    fprintf(file,"%f,", sumAverage);
    fprintf(file,"%f,", diferenceEntropy);
    fprintf(file,"%f\n", diferenceVariance);
    fclose(file);
}
int main(){
    clock_t Tmp[2];
    abrirFile();
    abrirFileTime();
    Mat image;
    int x=0;
    while (x!=200) {
        Tmp[0]=clock();
        x++;
        
        char add[200];
        sprintf(add, "/Users/wellcome/Desktop/BancoDeImagens/s (%d).png",x);
        image= imread(add,IMREAD_COLOR);
        cvtColor(image, image, CV_RGB2GRAY);
        
        glcm(image);
        
        Tmp[1]=clock();
        double Time= (Tmp[1]-Tmp[0])*1000.0/CLOCKS_PER_SEC;//Em MiliSegundos
        tempo=fopen(addressTime, "a");
        fprintf(tempo, "%f\n",Time);
        fclose(tempo);
    }
}