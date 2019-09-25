#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;

FILE* file;
char address[500]="/Users/wellcome/Desktop/DataBase/numbersDB/num_padronizado_MomCent_OpenCv.txt";//Banco de dados - Carregar
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
void leave_on_out(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,Mat Treino,Mat Teste,Mat TreinoLabel, Mat TesteLabel){
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
                            TreinoLabel.at<float>(contadorTreino,xx)=label.at<float>(y);
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
                            TesteLabel.at<float>(contadorTeste,xx)=label.at<float>(y);
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
void hold_out(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,Mat Treino,Mat Teste,Mat TreinoLabel,Mat TesteLabel){
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
                            TreinoLabel.at<float>(contadorTreino,xx)=labelHold.at<float>(y);
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
                            TesteLabel.at<float>(contadorTeste,xx)=labelHold.at<float>(y);
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
void ProgramToFile(Mat confusion,int quantidadeDeClasses,float atributos,float objetos,float acerto,float erro, int kernell,int decision,float perTreino,float perTeste){
    file=fopen(Resultados, "w");
    if (kernell==1) {
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : SVM\n\t\tKERNELL LINEAR\n");
    }
    if (kernell==2) {
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : SVM\n\t\tKERNELL RBF\n");
    }
    if (kernell==3) {
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : SVM\n\t\tKERNELL POLY\n");
    }
    if (kernell==4) {
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : SVM\n\t\tKERNELL SIGMOID\n");
    }
    if (decision==1) {
        fprintf(file,"\t\tMETODO DE SEPARACAO DE MATRIZES : LEAVE_ON_OUT\n");
    }
    if (decision==2) {
        fprintf(file,"\t\tMETODO DE SEPARACAO DE MATRIZES: HOLD_OUT\n");
    }
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
void Method_SVM(Mat Treino,Mat Teste, Mat TreinoLabel, Mat TesteLabel,Mat atrib, Mat label, float objetos, float atributos,int quantidadeClasses,int NumObjPorClasse[quantidadeClasses], float perTreino, float perTeste,Mat Confusion){
    //Escolher: Leave_on_out or Hold_out(Metodos)
    int decision;
    printf("Selecione:\n1-Leave_On_Out\n2-Hold-out\n");
    scanf("%d",&decision);
    if (decision==1) {
        leave_on_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel, TesteLabel);
    }
    if (decision==2) {
        hold_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel, TesteLabel);
    }
    if (decision!=1 && decision!=2) {
        printf("Opcao invalida!");
        return;
    }
    
    //SVM
    printf("Bando de dados: %s\n",address);
    printf("Metodo de Classificacao SVM\n");
    
    //------------------------------------------------------------
    //-----------------SVM PARAMETROS-----------------------------
    //------------------------------------------------------------
    
    int kernell;
    //int kernell = 1; //Roda a SVM com kernell Linear
    //int kernell = 2; //Roda a SVM com kernell RBF
    //int kernell = 3; //Roda a SVM com kernell Poly
    //int kernell = 4; //Roda a SVM com kernell Sigmoid
    printf("Digite o Kernell que deseja rodar no SVM:\n1 - kernell Linear\n2 - kernell RBF\n3 - kernell Poly\n4 - kernell Sigmoid\n");
    scanf("%d",&kernell);
    
    CvSVMParams param = CvSVMParams();
    
    param.svm_type = CvSVM::C_SVC;
    
    switch (kernell) {
        case 1:
            param.kernel_type = CvSVM::LINEAR;
            break;
        case 2:
            param.kernel_type = CvSVM::RBF;
            break;
        case 3:
            param.kernel_type = CvSVM::POLY;
            break;
        case 4:
            param.kernel_type = CvSVM::SIGMOID;
            break;
        default:
            param.kernel_type = CvSVM::LINEAR;
            break;
    }
    
    param.degree = 1; // for poly
    param.gamma = 20; // for poly/rbf/sigmoid
    param.coef0 =10; // for poly/sigmoid
    
    param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    param.p = 0.0; // for CV_SVM_EPS_SVR
    
    param.class_weights = NULL; // for CV_SVM_C_SVC
    param.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
    param.term_crit.max_iter = 1000;
    param.term_crit.epsilon = 1e-6;
    
    
    //------------------------------------------------------------
    //-----------------SVM TREINAMENTO----------------------------
    //------------------------------------------------------------
    
    CvSVM SVM;
    
    printf( "\nSVM training using dataset...");
    
    int iteractions = SVM.train(Treino, TreinoLabel, cv::Mat(), cv::Mat(), param);
    
    printf( "OK\n\n%d Iteracoes", iteractions);
    
    
    
    //------------------------------------------------------------
    //-----------------SVM CLASSIFICACAO--------------------------
    //------------------------------------------------------------
    
    Mat test_sample;
    
    float acerto=0;
    float erro=0;
    
    for (int tsample = 0; tsample < ((int)objetos*(perTeste/100)-1); tsample++) {
        
        test_sample=Teste.row(tsample);
        
        int res = (int)(SVM.predict(test_sample));
        int test = (int)(TesteLabel.at<float>(tsample));
        
        
        printf("Testing Sample %i -> class result (digit %d\t%d)\n", tsample, res, test);
        
        if (test!=res){
            erro++;
            Confusion.at<float>(test,res)++;
        }
        else{
            acerto++;
            Confusion.at<float>(test,res)++;
        }
        
    }
    printf("Taxa de Acerto : %0.001f %%",((acerto*100)/(acerto+erro)));
    ProgramToFile(Confusion, quantidadeClasses, atributos, objetos, acerto, erro, kernell, decision, perTreino, perTeste);
}
int main(){
    float objetos,atributos;
    
    //Pegar Quantidade de Objetos e Atributos do Banco de Dados
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
    
    //Atribuir valores as matrizes "Atrib" e "Label"
    Mat atrib(objetos,atributos,CV_32FC1);
    Mat label(objetos,1,CV_32FC1);
    AtribuirValoresMatrizes(BancoDeDados,atrib,label, objetos, atributos);
    
    //Pegar quantidade de classes
    int quantidadeClasses;
    quantidadeClasses=AcharQuantidadeDeClasses(quantidadeClasses, label, objetos);
    quantidadeClasses++;//Incrementar pois a quantidade de classes vai de 0 ate o maior e eu achei o maior.
    printf("Quantidade de Classes: %d\n",quantidadeClasses);
    
    //Pegar Quantidade de Objetos Por Classe
    float perTreino,perTeste;
    int NumObjPorClasse[quantidadeClasses];
    PegarQuantidadeDeObjetoPorClasse(quantidadeClasses,NumObjPorClasse,objetos,label);
    
    //Separando percentual de Treino e teste
    printf("Digite o percentual de treino: ");
    scanf("%f",&perTreino);
    perTeste=100-perTreino;
    
    //Declaracao de Matrizes Treino e Teste
    Mat Treino((objetos*(perTreino/100)),atributos,CV_32FC1);
    Mat TreinoLabel((objetos*(perTreino/100)),1,CV_32FC1);
    Mat Teste((objetos*(perTeste/100)),atributos,CV_32FC1);
    Mat TesteLabel((objetos*(perTeste/100)),1,CV_32FC1);
    
    Mat Confusion(quantidadeClasses,quantidadeClasses,CV_32FC1);
    for (int a=0; a<quantidadeClasses; a++) {
        for (int b=0; b<quantidadeClasses; b++) {
            Confusion.at<float>(a,b)=0;
        }
    }
    Method_SVM(Treino, Teste, TreinoLabel, TesteLabel, atrib, label, objetos, atributos, quantidadeClasses, NumObjPorClasse, perTreino, perTeste, Confusion);
}