//
//  main.cpp
//  IA
//
//  Created by Wellington Mendes on 21/08/15.
//  Copyright (c) 2015 Wellington Mendes. All rights reserved.
//

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
using namespace cv;

FILE* file;
char address[500]="/Users/wellcome/Desktop/DataBase/numbersDB/num_padronizado_MomStat_OpenCv.txt";//Banco de dados - Carregar
char addressSave[500]="/Users/wellcome/Desktop/DataBase/DataBase_Normalized.txt";//Banco de dados normalizado - Salvar
char Resultados[500]="/Users/wellcome/Desktop/Results.txt";//Resultados

void GetObjetosAtributos(FILE*file,float &objetos,float &atributos){
    objetos=0;
    atributos=0;
    float cont=-1;
    float num;
    char string[5000];
    file=fopen(address, "r");
    while (!feof(file)){
        fgets(string, 5000, file);
        objetos++;
    }
    objetos--;
    fclose(file);
    file=fopen(address, "r");
    while (!feof(file)) {
        fscanf(file,"%f,",&num);
        cont++;
    }
    fclose(file);
    atributos=cont/objetos;
    atributos--;
    atributos=int(atributos);
    objetos=int(objetos);
}
void PassarBancoDeDadosParaMat(Mat BancoDeDados,float objetos,float atributos){
    file=fopen(address, "r");
    float num;
    for (int obj=0; obj<objetos; obj++) {
        for (int atrib=0; atrib<atributos+1; atrib++) {
            if (atrib!=atributos) {
                fscanf(file,"%f,",&num);
                BancoDeDados.at<float>(obj,atrib)=num;
            }
            if (atrib==atributos) {
                fscanf(file,"%f",&num);
                BancoDeDados.at<float>(obj,atrib)=num;
            }
        }
    }
    fclose(file);
}
void Normalizar(Mat BancoDeDados,float objetos,float atributos){
    //Criacao de variaveis de armazenamento
    float** matNormalizar;
    float ma,me;
    matNormalizar= new float*[int(atributos)];
    for (int ind=0; ind<atributos; ind++) {
        matNormalizar[ind]=new float[2];
    }
    for (int ind1=0; ind1<atributos; ind1++) {
        for (int ind2=0; ind2<2; ind2++) {
            matNormalizar[ind1][ind2]=0;
        }
    }
    //Extracao do maior e menor de cada atributo
    for (int atrib=0; atrib<atributos; atrib++){
        for (int obj=0; obj<objetos; obj++) {
            if(obj==0){
                ma=BancoDeDados.at<float>(obj,atrib);
                me=BancoDeDados.at<float>(obj,atrib);
            }
            else{
                if (ma<BancoDeDados.at<float>(obj,atrib)) {
                    ma=BancoDeDados.at<float>(obj,atrib);
                }
                if (me>BancoDeDados.at<float>(obj,atrib)) {
                    me=BancoDeDados.at<float>(obj,atrib);
                }
            }
        }
        matNormalizar[atrib][0]=me;
        matNormalizar[atrib][1]=ma;
    }
    //Normalizacao
    for (int atrib=0; atrib<atributos; atrib++) {
        for (int obj=0; obj<objetos; obj++) {
            BancoDeDados.at<float>(obj,atrib)=(BancoDeDados.at<float>(obj,atrib)-matNormalizar[atrib][0])/(matNormalizar[atrib][1]-matNormalizar[atrib][0]);
        }
    }
    //Salvar BancoDeDadosNormalizado
    file=fopen(addressSave, "w");
    for (int obj=0; obj<objetos; obj++) {
        for (int atrib=0; atrib<atributos+1; atrib++) {
            if(atrib!=atributos){
                fprintf(file, "%f,",BancoDeDados.at<float>(obj,atrib));
            }
            if(atrib==atributos) {
                fprintf(file, "%1.f",BancoDeDados.at<float>(obj,atrib));
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
void AtribuirValoresMatrizes(Mat BancoDeDados,Mat atributos,Mat labels,float objetos,float atributo){
    atributo++;
    float num;
    for (int x=0; x<objetos; x++){
        for (int y=0;y<atributo; y++){
            if (y==atributo-1) {
                labels.at<float>(x,0)=BancoDeDados.at<float>(x,y);
            }
            else{
                atributos.at<float>(x,y)=BancoDeDados.at<float>(x,y);
            }
        }
    }
}
int AcharQuantidadeDeClasses(int quantidade,Mat label,float objetos){
    quantidade=0;
    for (int i=0; i<objetos; i++) {
        if ((label.at<float>(i))>(quantidade)) {
            quantidade=label.at<float>(i);
        }
    }
    return quantidade;
}
void PegarQuantidadeDeObjetoPorClasse(int quantidadeClasses,int numObjPorClasses[quantidadeClasses],int objetos,Mat label){
    for (int x=0; x<quantidadeClasses; x++){
        numObjPorClasses[x]=-1;
    }
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<objetos; y++) {
            if (label.at<float>(y)==x) {
                numObjPorClasses[x]++;
            }
        }
    }
}
int conferir_KNN5(int posMe,float vetPosMe[5],float vetResults[5]){
    if (vetResults[0]!=vetResults[1]!=vetResults[2]!=vetResults[3]!=vetResults[4])
        posMe=vetPosMe[0];
    float* vetRepet=new float[5];
    for (int rep=0; rep<5; rep++) {
        vetRepet[rep]=0;
    }
    for (int x=0; x<5; x++) {
        for (int ind=0; ind<5; ind++) {
            if (vetResults[x]==vetResults[ind] && x!=ind) {
                vetRepet[x]++;
            }
        }
    }
    int ma=0;
    for (int x=0; x<5; x++) {
        if (ma<vetRepet[x]) {
            posMe=vetPosMe[x];
        }
    }
    return posMe;
}
void leave_on_out(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,Mat Treino,Mat Teste){
    int contadorTreino=-1;
    int contadorTeste=-1;
    for (int controle=0; controle<quantidadeDeClasses; controle++) {
        int contador=-1;
        for (int y=0; y<objetos; y++) {
            if (label.at<float>(y)==controle) {
                contador++;
                if(contador<=int((NumObjPorClasse[controle])*(percentualTreino/100))){
                    contadorTreino++;
                    for (int xx=0; xx<atributos+1; xx++){
                        if (xx!=atributos) {
                            Treino.at<float>(contadorTreino,xx)=atrib.at<float>(y,xx);
                        }
                        if (xx==atributos) {
                            Treino.at<float>(contadorTreino,xx)=label.at<float>(y);
                        }
                        //Mostrador de Separacao de Matrizes Treino
                        printf("\nClasse - %d\n",controle);
                        printf("%d ATE ",contador);
                        printf("%d\n",NumObjPorClasse[controle]);
                        printf("TREINO - %f\n",Treino.at<float>(contadorTreino,xx));
                    }
                }
                else{
                    contadorTeste++;
                    for (int xx=0; xx<atributos+1; xx++){
                        if (xx!=atributos) {
                            Teste.at<float>(contadorTeste,xx)=atrib.at<float>(y,xx);
                        }
                        if (xx==atributos) {
                            Teste.at<float>(contadorTeste,xx)=label.at<float>(y);
                        }
                        //Mostrador de Separacao de Matrizes Teste
                        printf("\nClasse - %d\n",controle);
                        printf("%d ATE ",contador);
                        printf("%d\n",NumObjPorClasse[controle]);
                        printf("TESTE - %f\n",Teste.at<float>(contadorTeste,xx));
                    }
                }
            }
        }
    }
}
void hold_out(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,Mat Treino,Mat Teste){
    int y;
    int contadorTreino=-1;
    int contadorTeste=-1;
    for (int controle=0; controle<quantidadeDeClasses; controle++){
        Mat atribHold(NumObjPorClasse[controle],atributos,CV_32FC1);
        Mat labelHold(NumObjPorClasse[controle],1,CV_32FC1);
        Mat vetObj(NumObjPorClasse[controle],1,CV_32FC1);
        int por=-1;
        for (int x=0; x<objetos; x++) {
            if (label.at<float>(x)==controle) {
                por++;
                for (int z=0; z<atributos+1; z++) {
                    if (z!=atributos) {
                        atribHold.at<float>(por,z)=atrib.at<float>(x,z);
                    }
                    if (z==atributos) {
                        labelHold.at<float>(por)=label.at<float>(x);
                    }
                }
            }
        }
        for (int x=0; x<NumObjPorClasse[controle]; x++) {
            vetObj.at<float>(x)=labelHold.at<float>(x);
        }
        int objetovar=int(NumObjPorClasse[controle]);
        int contador=0;
        srand(time(NULL));
        while (contador!=NumObjPorClasse[controle]) {
            y=rand()%objetovar;
            if (vetObj.at<float>(y)!=-1){
                vetObj.at<float>(y)=-1;
                contador++;
                if(contador<=int((NumObjPorClasse[controle])*(percentualTreino/100))){
                    contadorTreino++;
                    for (int xx=0; xx<atributos+1; xx++){
                        if (xx!=atributos) {
                            Treino.at<float>(contadorTreino,xx)=atribHold.at<float>(y,xx);
                        }
                        if (xx==atributos) {
                            Treino.at<float>(contadorTreino,xx)=labelHold.at<float>(y);
                        }
                        //Mostrador de Separacao de Matrizes Treino
                        printf("\nClasse - %d\n",controle);
                        printf("%d ATE ",contador);
                        printf("%d\n",NumObjPorClasse[controle]);
                        printf("TREINO - %f\n",Treino.at<float>(contadorTreino,xx));
                    }
                }
                else{
                    contadorTeste++;
                    for (int xx=0; xx<atributos+1; xx++){
                        if (xx!=atributos) {
                            Teste.at<float>(contadorTeste,xx)=atribHold.at<float>(y,xx);
                        }
                        if (xx==atributos) {
                            Teste.at<float>(contadorTeste,xx)=labelHold.at<float>(y);
                        }
                        //Mostrador de Separacao de Matrizes Teste
                        printf("\nClasse - %d\n",controle);
                        printf("%d ATE ",contador);
                        printf("%d\n",NumObjPorClasse[controle]);
                        printf("TESTE - %f\n",Teste.at<float>(contadorTeste,xx));
                    }
                }
            }
        }
    }
}
void hold_out_other(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,Mat Treino,Mat Teste){
    int y;
    int contadorTreino=-1;
    int contadorTeste=-1;
    for (int controle=0; controle<quantidadeDeClasses; controle++){
        Mat vetObj(NumObjPorClasse[controle],1,CV_32FC1);
        for (int x=0; x<objetos; x++) {
            vetObj.at<float>(x)=label.at<float>(x);
        }
        int objetovar=int(objetos);
        int contador=0;
        srand(time(NULL));
        while (contador!=NumObjPorClasse[controle]) {
            y=rand()%objetovar;
            if (vetObj.at<float>(y)==controle && vetObj.at<float>(y)!=-1){
                vetObj.at<float>(y)=-1;
                contador++;
                if(contador<=int((NumObjPorClasse[controle])*(percentualTreino/100))){
                    contadorTreino++;
                    for (int xx=0; xx<atributos+1; xx++){
                        if (xx!=atributos) {
                            Treino.at<float>(contadorTreino,xx)=atrib.at<float>(y,xx);
                        }
                        if (xx==atributos) {
                            Treino.at<float>(contadorTreino,xx)=label.at<float>(y);
                        }
                        //Mostrador de Separacao de Matrizes Treino
                        printf("\nClasse - %d\n",controle);
                        printf("%d ATE ",contador);
                        printf("%d\n",NumObjPorClasse[controle]);
                        printf("TREINO - %f\n",Treino.at<float>(contadorTreino,xx));
                    }
                }
                else{
                    contadorTeste++;
                    for (int xx=0; xx<atributos+1; xx++){
                        if (xx!=atributos) {
                            Teste.at<float>(contadorTeste,xx)=atrib.at<float>(y,xx);
                        }
                        if (xx==atributos) {
                            Teste.at<float>(contadorTeste,xx)=label.at<float>(y);
                        }
                        //Mostrador de Separacao de Matrizes Teste
                        printf("\nClasse - %d\n",controle);
                        printf("%d ATE ",contador);
                        printf("%d\n",NumObjPorClasse[controle]);
                        printf("TESTE - %f\n",Teste.at<float>(contadorTeste,xx));
                    }
                }
            }
        }
    }
}
void ProgramToFile(Mat confusion,int quantidadeDeClasses,float atributos,float objetos,float acerto,float erro,int decision,float perTreino,float perTeste,int K){
    file=fopen(Resultados, "w");
    if (K==1)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : KNN 1\n");
    if (K==3)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : KNN 3\n");
    if (K==5)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : KNN 5\n");
    if (K==0)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : KMEANS\n");
    if (decision==1)
        fprintf(file,"\t\tMETODO DE SEPARACAO DE MATRIZES : LEAVE_ON_OUT\n");
    if (decision==2)
        fprintf(file,"\t\tMETODO DE SEPARACAO DE MATRIZES: HOLD_OUT\n");
    fprintf(file, "OBJETOS - %1.f\nATRIBUTOS - %1.f\nCLASSES - %d\nPORCENTAGEM DE TREINO - %1.f %%\nPORCENTAGEM DE TESTE - %1.f %%\n",atributos,objetos,quantidadeDeClasses,perTreino,perTeste);
    fprintf(file,"TAXA DE ACERTO : %0.001f %%\n",((acerto*100)/(acerto+erro)));
    fprintf(file,"\t\t\tMATRIZ DE CONFUSAO\n");
    for (int i=0; i<quantidadeDeClasses; i++) {
        if (i==0)
            fprintf(file, "\t");
        fprintf(file,"%d\t",i);
        if (i==quantidadeDeClasses-1)
            fprintf(file, "\n\n");
    }
    for (int x=0; x<quantidadeDeClasses; x++) {
        for (int y=0; y<quantidadeDeClasses; y++) {
            if (y==0) {
                fprintf(file, "%d\t",x);
            }
            fprintf(file,"%1.f\t",confusion.at<float>(x,y));
        }
        fprintf(file,"\n");
    }
    fclose(file);
}
void KNN1(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,int decision,Mat Confusion){
    int K=1;
    float acerto=0,erro=0;
    Mat Treino((objetos*(percentualTreino/100)),atributos+1,CV_32FC1);
    Mat Teste((objetos*(percentualTeste/100)),atributos+1,CV_32FC1);
    if (decision==1) {
        leave_on_out(percentualTreino, percentualTeste, quantidadeDeClasses, NumObjPorClasse, atributos, objetos, label, atrib,Treino,Teste);
    }
    if (decision==2) {
        hold_out(percentualTreino, percentualTeste, quantidadeDeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste);
    }
    //************KNN1************//
    float* resultTest= new float [int((objetos*percentualTreino/100))];
    //ExtraindoResultadosDeCadaClasse
    
    for (int matTest=0; matTest<(int(objetos*(percentualTeste/100))); matTest++){
        for (int matTreino=0; matTreino<(int(objetos*(percentualTreino/100))); matTreino++) {
            float resultsLocal=0;
            for (int varrerAtributos=0; varrerAtributos<atributos; varrerAtributos++) {
                resultsLocal+=pow(Teste.at<float>(matTest,varrerAtributos)-Treino.at<float>(matTreino,varrerAtributos),2);
            }
            resultTest[matTreino]=sqrt(resultsLocal);
        }
        //Achar o Menor e achar posicao do menor
        float me=1;
        int posMe=0;
        for (int menor=0; menor<int(((percentualTreino/100)*objetos)); menor++) {
            if (me>resultTest[menor]) {
                me=resultTest[menor];
                posMe=menor;
            }
        }
        //Conferir
        //printf("Treino - %f // Teste - %f\n",Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos));
        if (Treino.at<float>(posMe,atributos)==Teste.at<float>(matTest,atributos)) {
            acerto++;
            Confusion.at<float>(Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos))++;
        }
        else{
            erro++;
            Confusion.at<float>(Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos))++;
        }
        printf("\n%d - %d\n",matTest,(int(objetos*(percentualTeste/100)))-1);
    }
    delete [] resultTest;
    printf("\nTaxa De Acerto - %0.001f%%\n",((acerto*100)/(acerto+erro)));
    printf("Acerto - %f\nErro - %f\n",acerto,erro);
    printf("\n_Fim_Do_KNN1_\n");
    ProgramToFile(Confusion, quantidadeDeClasses, atributos, objetos, acerto, erro, decision, percentualTreino, percentualTeste,K);
}
void KNN3(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,int decision,Mat Confusion){
    int K=3;
    float acerto=0,erro=0;
    Mat Treino((objetos*(percentualTreino/100)),atributos+1,CV_32FC1);
    Mat Teste((objetos*(percentualTeste/100)),atributos+1,CV_32FC1);
    if (decision==1) {
        leave_on_out(percentualTreino, percentualTeste, quantidadeDeClasses, NumObjPorClasse, atributos, objetos, label, atrib,Treino,Teste);
    }
    if (decision==2) {
        hold_out(percentualTreino, percentualTeste, quantidadeDeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste);
    }
    //************KNN3************//
    float* resultTest= new float [int((objetos*percentualTreino/100))];
    float* vetResults= new float [3];
    float* vetPosMe= new float[3];
    //ExtraindoResultadosDeCadaClasse
    
    for (int matTest=0; matTest<(int(objetos*(percentualTeste/100))); matTest++){
        for (int matTreino=0; matTreino<(int(objetos*(percentualTreino/100))); matTreino++) {
            float resultsLocal=0;
            for (int varrerAtributos=0; varrerAtributos<atributos; varrerAtributos++) {
                resultsLocal+=pow(Teste.at<float>(matTest,varrerAtributos)-Treino.at<float>(matTreino,varrerAtributos),2);
            }
            resultTest[matTreino]=sqrt(resultsLocal);
        }
        //Achar o Menor e achar posicao do menor
        float me=10;
        int posMe=0;
        for (int ind=0; ind<3; ind++) {
            me=10;
            posMe=0;
            for (int menor=0; menor<int(((percentualTreino/100)*objetos)); menor++) {
                if (me>resultTest[menor]) {
                    me=resultTest[menor];
                    posMe=menor;
                }
            }
            vetResults[ind]=me;
            vetPosMe[ind]=posMe;
            resultTest[posMe]=10;
        }
        //Achar o PosMe dos 3
        for (int ind=0; ind<3; ind++) {
            vetResults[ind]=Treino.at<float>(vetPosMe[ind],atributos);
        }
        for (int ind=0; ind<1; ind++) {
            if (vetResults[0]!=vetResults[1]!=vetResults[2]) {
                posMe=vetPosMe[0];
            }
            else if(vetResults[0]==vetResults[1]){
                posMe=vetPosMe[0];
            }
            else if(vetResults[1]==vetResults[2]){
                posMe=vetPosMe[1];
            }
            else if(vetResults[2]==vetResults[0]){
                posMe=vetPosMe[2];
            }
        }
        //Conferir
        //printf("Treino - %f // Teste - %f\n",Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos));
        if (Treino.at<float>(posMe,atributos)==Teste.at<float>(matTest,atributos)) {
            acerto++;
            Confusion.at<float>(Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos))++;
        }
        else{
            erro++;
            Confusion.at<float>(Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos))++;
        }
        printf("\n%d - %d\n",matTest,(int(objetos*(percentualTeste/100)))-1);
    }
    delete [] resultTest;
    delete [] vetPosMe;
    delete [] vetResults;
    printf("\nTaxa De Acerto - %0.001f%%\n",((acerto*100)/(acerto+erro)));
    printf("Acerto - %f\nErro - %f\n",acerto,erro);
    printf("\n_Fim_Do_KNN3_\n");
    ProgramToFile(Confusion, quantidadeDeClasses, atributos, objetos, acerto, erro, decision, percentualTreino, percentualTeste,K);
}
void KNN5(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,int decision,Mat Confusion){
    int K=5;
    float acerto=0,erro=0;
    Mat Treino((objetos*(percentualTreino/100)),atributos+1,CV_32FC1);
    Mat Teste((objetos*(percentualTeste/100)),atributos+1,CV_32FC1);
    if (decision==1) {
        leave_on_out(percentualTreino, percentualTeste, quantidadeDeClasses, NumObjPorClasse, atributos, objetos, label, atrib,Treino,Teste);
    }
    if (decision==2) {
        hold_out(percentualTreino, percentualTeste, quantidadeDeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste);
    }
    //************KNN5************//
    float* resultTest= new float [int((objetos*percentualTreino/100))];
    float* vetResults= new float [5];
    float* vetPosMe= new float[5];
    //ExtraindoResultadosDeCadaClasse
    
    for (int matTest=0; matTest<(int(objetos*(percentualTeste/100))); matTest++){
        for (int matTreino=0; matTreino<(int(objetos*(percentualTreino/100))); matTreino++) {
            float resultsLocal=0;
            for (int varrerAtributos=0; varrerAtributos<atributos; varrerAtributos++) {
                resultsLocal+=pow(Teste.at<float>(matTest,varrerAtributos)-Treino.at<float>(matTreino,varrerAtributos),2);
            }
            resultTest[matTreino]=sqrt(resultsLocal);
        }
        //Achar o Menor e achar posicao do menor
        float me=10;
        int posMe=0;
        for (int ind=0; ind<5; ind++) {
            me=10;
            posMe=0;
            for (int menor=0; menor<int(((percentualTreino/100)*objetos)); menor++) {
                if (me>resultTest[menor]) {
                    me=resultTest[menor];
                    posMe=menor;
                }
            }
            vetResults[ind]=me;
            vetPosMe[ind]=posMe;
            resultTest[posMe]=10;
        }
        //Achar o PosMe dos 5
        for (int ind=0; ind<5; ind++) {
            vetResults[ind]=Treino.at<float>(vetPosMe[ind],atributos);
        }
        posMe=conferir_KNN5(posMe, vetPosMe, vetResults);
        
        //Conferir
        //printf("Treino - %f // Teste - %f\n",Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos));
        if (Treino.at<float>(posMe,atributos)==Teste.at<float>(matTest,atributos)) {
            acerto++;
            Confusion.at<float>(Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos))++;
        }
        else{
            erro++;
            Confusion.at<float>(Treino.at<float>(posMe,atributos),Teste.at<float>(matTest,atributos))++;
        }
        printf("\n%d - %d\n",matTest,(int(objetos*(percentualTeste/100)))-1);
    }
    delete [] resultTest;
    delete [] vetPosMe;
    delete [] vetResults;
    printf("\nTaxa De Acerto - %0.001f%%\n",((acerto*100)/(acerto+erro)));
    printf("Acerto - %f\nErro - %f\n",acerto,erro);
    printf("\n_Fim_Do_KNN5_\n");
    ProgramToFile(Confusion, quantidadeDeClasses, atributos, objetos, acerto, erro, decision, percentualTreino, percentualTeste,K);
}
void KMEANS(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,int decision, Mat Confusion){
    int K=0;
    float acerto=0,erro=0;
    Mat Treino((objetos*(percentualTreino/100)),atributos+1,CV_32FC1);
    Mat Teste((objetos*(percentualTeste/100)),atributos+1,CV_32FC1);
    if (decision==1) {
        leave_on_out(percentualTreino, percentualTeste, quantidadeDeClasses, NumObjPorClasse, atributos, objetos, label, atrib,Treino,Teste);
    }
    if (decision==2){
        hold_out(percentualTreino, percentualTeste, quantidadeDeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste);
    }
    //************KMEANS************//
    float* resultTest= new float [quantidadeDeClasses];
    
    //Calculando a media na Mat Treino
    Mat TreinoKMeans(quantidadeDeClasses,atributos,CV_32FC1);
    for (int lbb=0; lbb<quantidadeDeClasses; lbb++) {
        for (int att=0; att<atributos; att++){
            float means=0;
            int cont=0;
            for (int objj=0; objj<int((objetos*percentualTreino/100)); objj++) {
                if (Treino.at<float>(objj,atributos)==lbb) {
                    means+=Treino.at<float>(objj,att);
                    cont++;
                }
            }
            means=means/cont;
            TreinoKMeans.at<float>(lbb,att)=means;
        }
    }
    //ExtraindoResultadosDeCadaClasse
    for (int matTest=0; matTest<(int(objetos*(percentualTeste/100))); matTest++){
        for (int matTreino=0; matTreino<quantidadeDeClasses; matTreino++) {
            float resultsLocal=0;
            for (int varrerAtributos=0; varrerAtributos<atributos; varrerAtributos++) {
                resultsLocal+=pow(Teste.at<float>(matTest,varrerAtributos)-TreinoKMeans.at<float>(matTreino,varrerAtributos),2);
            }
            resultTest[matTreino]=sqrt(resultsLocal);
        }
        //Achar o Menor e achar posicao do menor
        float me=10;
        int posMe=0;
        for (int menor=0; menor<quantidadeDeClasses; menor++) {
            if (me>resultTest[menor]) {
                me=resultTest[menor];
                posMe=menor;
            }
        }
        //Conferir
        //printf("Treino - %d // Teste - %1.f\n",posMe,Teste.at<float>(matTest,atributos));
        if (posMe==Teste.at<float>(matTest,atributos)) {
            acerto++;
            Confusion.at<float>(posMe,Teste.at<float>(matTest,atributos))++;
        }
        else{
            erro++;
            Confusion.at<float>(posMe,Teste.at<float>(matTest,atributos))++;
        }
        printf("\n%d - %d\n",matTest,(int(objetos*(percentualTeste/100)))-1);
    }
    delete [] resultTest;
    printf("\nTaxa De Acerto - %0.001f%%\n",((acerto*100)/(acerto+erro)));
    printf("Acerto - %f\nErro - %f\n",acerto,erro);
    printf("\n_Fim_Do_KMEANS_\n");
    ProgramToFile(Confusion, quantidadeDeClasses, atributos, objetos, acerto, erro, decision, percentualTreino, percentualTeste,K);
}

int main(){
    int selecione;
    float objetos,atributos;
    int decision;
    GetObjetosAtributos(file, objetos, atributos);
    printf("Objetos - %1.f\nAtributos - %1.f\n",objetos,atributos);//Quantidade de objetos e atributos
    
    //Passar Banco de Dados Para Mat
    Mat BancoDeDados(objetos,atributos+1,CV_32FC1);
    PassarBancoDeDadosParaMat(BancoDeDados,objetos,atributos);
    
    //Normalizar Banco de Dados//
    int opc;
    printf("Deseja normalizar o banco de Dados?\n1-Sim\n2-Nao\n");
    scanf("%d",&opc);
    if (opc==1) {
        Normalizar(BancoDeDados,objetos,atributos);
    }
    
    Mat atrib(objetos,atributos,CV_32FC1);
    Mat label(objetos,1,CV_32FC1);
    AtribuirValoresMatrizes(BancoDeDados,atrib,label, objetos, atributos);
    
    int quantidadeClasses;
    quantidadeClasses=AcharQuantidadeDeClasses(quantidadeClasses, label, objetos);
    quantidadeClasses++;//Incrementar pois a quantidade de classes vai de 0 ate o maior e eu achei o maior.
    printf("Quantidade de Classes: %d\n",quantidadeClasses);
    
    float perTreino,perTeste;
    int NumObjPorClasse[quantidadeClasses];
    //Criar Mat Confusao
    Mat Confusion(quantidadeClasses,quantidadeClasses,CV_32FC1);
    for (int a=0; a<quantidadeClasses; a++) {
        for (int b=0; b<quantidadeClasses; b++) {
            Confusion.at<float>(a,b)=0;
        }
    }
    
    printf("Bando de dados: %s\n",address);
    printf("Selecione o Metodo de Classificacao:\n1-KNN1\n2-KNN3\n3-KNN5\n4-KMEANS\n");
    scanf("%d",&selecione);
    switch (selecione) {
        case 1:
            //Escolher: Leave_on_out or Hold_out
            printf("Selecione:\n1-Leave_On_Out\n2-Hold-out\n");
            scanf("%d",&decision);
            //Separando percentual de Treino e teste
            printf("Digite o percentual de treino: ");
            scanf("%f",&perTreino);
            perTeste=100-perTreino;
            //SepararMatrizTreinoETeste
            PegarQuantidadeDeObjetoPorClasse(quantidadeClasses,NumObjPorClasse,objetos,label);
            KNN1(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib,decision,Confusion);
            break;
        case 2:
            //Escolher: Leave_on_out or Hold_out
            printf("Selecione:\n1-Leave_On_Out\n2-Hold-out\n");
            scanf("%d",&decision);
            //Separando percentual de Treino e teste
            printf("Digite o percentual de treino: ");
            scanf("%f",&perTreino);
            perTeste=100-perTreino;
            //SepararMatrizTreinoETeste
            PegarQuantidadeDeObjetoPorClasse(quantidadeClasses,NumObjPorClasse,objetos,label);
            KNN3(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, decision,Confusion);
            break;
        case 3:
            //Escolher: Leave_on_out or Hold_out
            printf("Selecione:\n1-Leave_On_Out\n2-Hold-out\n");
            scanf("%d",&decision);
            //Separando percentual de Treino e teste
            printf("Digite o percentual de treino: ");
            scanf("%f",&perTreino);
            perTeste=100-perTreino;
            //SepararMatrizTreinoETeste
            PegarQuantidadeDeObjetoPorClasse(quantidadeClasses,NumObjPorClasse,objetos,label);
            KNN5(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, decision,Confusion);
            break;
        case 4:
            //Escolher: Leave_on_out or Hold_out
            printf("Selecione:\n1-Leave_On_Out\n2-Hold-out\n");
            scanf("%d",&decision);
            //Separando percentual de Treino e teste
            printf("Digite o percentual de treino: ");
            scanf("%f",&perTreino);
            perTeste=100-perTreino;
            //SepararMatrizTreinoETeste
            PegarQuantidadeDeObjetoPorClasse(quantidadeClasses,NumObjPorClasse,objetos,label);
            KMEANS(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, decision,Confusion);
            break;
        default:
            printf("\nOpçao Invalida!\n");
            break;
    }
}