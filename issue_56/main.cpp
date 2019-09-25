#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/ml.h>
#include <stdlib.h>
#include <stdio.h>
using namespace cv;

FILE* file;
char address[500]="/Users/wellcome/Desktop/DataBase/numbersDB/num_padronizado_MomStat_OpenCv.txt";//Banco de dados - Carregar
char addressSave[500]="/Users/wellcome/Desktop/DataBase/DataBase_Normalized.txt";//Banco de dados normalizado - Salvar

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
void Method_MLP(Mat Treino,Mat Teste,Mat TreinoLabel,Mat TesteLabel,Mat label,Mat atrib,float perTreino,float perTeste,int quantidadeClasses,int NumObjPorClasse[quantidadeClasses],float atributos,float objetos){
    int decision;
    //Escolher: Leave_on_out or Hold_out(Metodos)
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
    Mat teste_lable_MLP((int)(objetos*(perTeste/100)),quantidadeClasses,CV_32FC1);
    Mat treino_lable_MLP((int)(objetos*(perTreino/100)),quantidadeClasses,CV_32FC1);
    int x,y;
    int val;
    
    for(y=0;y<(int)(objetos*(perTeste/100));y++){
        val = (int)TesteLabel.at<float>(y,0);
        for(x=0;x<quantidadeClasses;x++){
            if(val==x){
                teste_lable_MLP.at<float>(y,x) = 1;
            }
            else{
                teste_lable_MLP.at<float>(y,x) = 0;
            }
        }
    }
    for(y=0;y<(int)(objetos*(perTreino/100));y++){
        val = (int)TreinoLabel.at<float>(y,0);
        for(x=0;x<quantidadeClasses;x++){
            if(val==x){
                treino_lable_MLP.at<float>(y,x) = 1;
            }
            else{
                treino_lable_MLP.at<float>(y,x) = 0;
            }
        }
    }
    
    //------------------------------------------------------------
    //--------------------------MLP-------------------------------
    //------------------------------------------------------------
    
    printf("Bando de dados: %s\n",address);
    printf("Metodo de Classificacao MLP\n");
    
    cv::Mat classificationResult(1, quantidadeClasses, CV_32FC1);
    
    
    //------------------------------------------------------------
    //-----------------MLP TREINAMENTO----------------------------
    //------------------------------------------------------------
    
    
    cv::Mat layers(3,1,CV_32S);
    
    int numHiddenLayer;
    int config;
    //int config = 1; //Roda a configuracao 1 da MLP
    //int config = 2; //Roda a configuracao 2 da MLP
    printf("Selecione a configuracao do MLP:\n1 - Configuracao 1\n2 - Configuracao 2\n");
    scanf("%d",&config);
    switch (config) {
        case 1:
            numHiddenLayer = (quantidadeClasses+atributos)/2;
            layers.at<int>(0,0) = atributos;//input layer
            layers.at<int>(1,0)=numHiddenLayer;//hidden layer
            layers.at<int>(2,0) =quantidadeClasses;//output layer
            break;
        case 2:
            numHiddenLayer = (quantidadeClasses+atributos)*2/3;
            layers.at<int>(0,0) = atributos;//input layer
            layers.at<int>(1,0)=numHiddenLayer;//hidden layer
            layers.at<int>(2,0) =quantidadeClasses;//output layer
            break;
        default:
            printf("\nOpcao invalida\n");
            return;
            break;
    }
    CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,0.6,1);
    
    CvANN_MLP_TrainParams params(cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),CvANN_MLP_TrainParams::BACKPROP,0.1,0.1);
    
    printf( "\nUsing training dataset\n");
    int iterations = nnetwork.train(Treino, treino_lable_MLP,Mat(),Mat(),params);
    printf( "Training iterations: %i\n\n", iterations);
    
    //------------------------------------------------------------
    //-----------------MLP CLASSIFICACAO--------------------------
    //------------------------------------------------------------
    
    Mat test_sample;
    
    float acerto=0;
    float erro=0;
    
    for (int tsample = 0; tsample < ((int)objetos*(perTeste/100)-1); tsample++) {
        
        test_sample=Teste.row(tsample);
        
        nnetwork.predict(test_sample, classificationResult);
        
        int maxIndex = 0;
        float value=0.0f;
        float maxValue=classificationResult.at<float>(0,0);
        for(int index=1;index<quantidadeClasses;index++)
        {   value = classificationResult.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex=index;
            }
        }
        
        printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);
        
        if (teste_lable_MLP.at<float>(tsample, maxIndex)!=1.0f){
            erro++;
            for(int class_index=0;class_index<quantidadeClasses;class_index++){
                if(teste_lable_MLP.at<float>(tsample, class_index)==1.0f){
                    break;
                }
            }
        }
        else{
            acerto++;
        }
    }
    printf("Taxa de Acerto : %0.001f %%",((acerto*100)/(acerto+erro)));
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
    Mat Treino((int)(objetos*(perTreino/100)),atributos,CV_32FC1);
    Mat TreinoLabel((int)(objetos*(perTreino/100)),quantidadeClasses,CV_32FC1);
    Mat Teste((int)(objetos*(perTeste/100)),atributos,CV_32FC1);
    Mat TesteLabel((int)(objetos*(perTeste/100)),quantidadeClasses,CV_32FC1);
    
    Method_MLP(Treino, Teste, TreinoLabel, TesteLabel, label, atrib, perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos);
}

