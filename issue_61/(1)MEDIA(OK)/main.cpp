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
char Resultados[500]="/Users/wellcome/Desktop/Results.txt";//Resuldados
int rod=0,K=-1;

//MANUSEAR BANCO DE DADOS//
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

//PARA ARQUIVO//
void InitializeArquivo(){
    file=fopen(Resultados, "w");
    fprintf(file, "\t\tRESULTADOS\n\n");
    fclose(file);
}
void ProgramToFile(Mat confusion,int quantidadeClasses,float atributos,float objetos,float acerto,float erro,int decision,float perTreino,float perTeste,float speciticivy,float sensitivity,float ppv,float accuracy,Mat ResPorClasse,int rod,Mat ResMedia,int Media){
    file=fopen(Resultados, "a");
    if (rod==1)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : KNN 1\n");
    if (rod==2)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : KNN 3\n");
    if (rod==3)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : KNN 5\n");
    if (rod==4)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : KMEANS\n");
    if (rod==5)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : SVM LINEAR\n");
    if (rod==6)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : SVM RBF\n");
    if (rod==7)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : SVM POLY\n");
    if (rod==8)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : SVM SIGMOID\n");
    if (rod==9)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : BAYES\n");
    if (rod==10)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : MLP CONFIG 01\n");
    if (rod==11)
        fprintf(file, "\t\tMETODO DE CLASSIFICACAO : MLP CONFIG 02\n");
    fprintf(file, "\t\t------ITERACAO %d------\n",Media);
    if (decision==1)
        fprintf(file,"\t\tMETODO DE SEPARACAO DE MATRIZES : LEAVE_ON_OUT\n");
    if (decision==2)
        fprintf(file,"\t\tMETODO DE SEPARACAO DE MATRIZES: HOLD_OUT\n");
    fprintf(file, "DATABASE - %s\n",address);
    fprintf(file, "OBJETOS - %1.f\nATRIBUTOS - %1.f\nCLASSES - %d\nPORCENTAGEM DE TREINO - %1.f %%\nPORCENTAGEM DE TESTE - %1.f %%\n",objetos,atributos,quantidadeClasses,perTreino,perTeste);
    fprintf(file,"TAXA DE ACERTO : %0.001f %%\n",((acerto*100)/(acerto+erro)));
    fprintf(file,"\t\t\t\tMATRIZ DE CONFUSAO\n");
    for (int i=0; i<quantidadeClasses; i++) {
        if (i==0)
            fprintf(file, "\t");
        fprintf(file,"%d\t",i);
        if (i==quantidadeClasses-1)
            fprintf(file, "\n\n");
    }
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<quantidadeClasses; y++) {
            if (y==0) {
                fprintf(file, "%d\t",x);
            }
            fprintf(file,"%1.f\t",confusion.at<float>(x,y));
        }
        fprintf(file,"\n");
    }
    fprintf(file, "\n\t\tRESULTADOS DAS MEDIDAS ESTATISTICAS\n");
    fprintf(file, "\t\tMEDIDAS POR CLASSIFICADOR\n");
    fprintf(file, "\nSpeciticivy - %.2f\n",speciticivy);
    fprintf(file, "Sensitivity - %.2f\n",sensitivity);
    fprintf(file, "Accuracy - %.2f\n",accuracy);
    fprintf(file, "PPV - %.2f\n\n",ppv);
    fprintf(file, "\t\tMEDIDAS POR CLASSE\n");
    fprintf(file, "\tspe\tsen\tacc\tppv\n\n");
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<4; y++) {
            if (y==0) {
                fprintf(file, "%d\t",x);
            }
            fprintf(file,"%0.2f\t",ResPorClasse.at<float>(x,y));
        }
        fprintf(file,"\n");
    }
    //Media
//    Mat ResMedia (10,4,CV_32FC1);
    ResMedia.at<float>(Media,0)=speciticivy;
    ResMedia.at<float>(Media,1)=sensitivity;
    ResMedia.at<float>(Media,2)=accuracy;
    ResMedia.at<float>(Media,3)=ppv;
    if (Media==9) {
        fprintf(file, "\n-------------------------------------------------------------------------------------------\n");
        fprintf(file, "-------------------------------------------------------------------------------------------\n");
        speciticivy=0;
        sensitivity=0;
        accuracy=0;
        ppv=0;
        for (int x=0; x<10; x++) {
            speciticivy+=ResMedia.at<float>(x,0);
            sensitivity+=ResMedia.at<float>(x,1);
            accuracy+=ResMedia.at<float>(x,2);
            ppv+=ResMedia.at<float>(x,3);
        }
        fprintf(file, "\n\t\tMEDIA DOS RESULTADOS DAS MEDIDAS ESTATISTICAS\n");
        fprintf(file, "\nMedia Speciticivy - %.2f\n",speciticivy/10);
        fprintf(file, "Media Sensitivity - %.2f\n",sensitivity/10);
        fprintf(file, "Media Accuracy - %.2f\n",accuracy/10);
        fprintf(file, "Media PPV - %.2f\n\n",ppv/10);
        fprintf(file, "-------------------------------------------------------------------------------------------\n");
        fprintf(file, "-------------------------------------------------------------------------------------------\n");
        fprintf(file, "-------------------------------------------------------------------------------------------\n\n");
        
        //Reinicializando ResMedia
        for (int n1=0; n1<10; n1++)
            for (int n2=0; n2<4; n2++)
                ResMedia.at<float>(n1,n2)=0;
    }
    if (Media!=9) {
        fprintf(file, "\n-------------------------------------------------------------------------------------------\n");
    }
    fclose(file);
    for (int a=0; a<quantidadeClasses; a++) {
        for (int b=0; b<4; b++) {
            confusion.at<float>(a,b)=0;
        }
    }
}

//METODOS DE SEPARACAO DO BANCO DE DADOS//
void leave_on_out(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,Mat Treino,Mat Teste,Mat TreinoLabel, Mat TesteLabel){
    printf("\nLeave On Out...\n");
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
//                        printf("\nClasse - %d\n",controle);
//                        printf("%d ATE ",contador);
//                        printf("%d\n",NumObjPorClasse[controle]);
//                        printf("TREINO - %f\n",Treino.at<float>(contadorTreino,xx));
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
//                        printf("\nClasse - %d\n",controle);
//                        printf("%d ATE ",contador);
//                        printf("%d\n",NumObjPorClasse[controle]);
//                        printf("TESTE - %f\n",Teste.at<float>(contadorTeste,xx));
                    }
                }
            }
        }
    }
}
void hold_out(float percentualTreino,float percentualTeste,int quantidadeDeClasses,int NumObjPorClasse[quantidadeDeClasses],float atributos,float objetos,Mat label,Mat atrib,Mat Treino,Mat Teste,Mat TreinoLabel,Mat TesteLabel){
    printf("\nHold Out...\n");
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
//                        printf("\nClasse - %d\n",controle);
//                        printf("%d ATE ",contador);
//                        printf("%d\n",NumObjPorClasse[controle]);
//                        printf("TREINO - %f\n",Treino.at<float>(contadorTreino,xx));
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
//                        printf("\nClasse - %d\n",controle);
//                        printf("%d ATE ",contador);
//                        printf("%d\n",NumObjPorClasse[controle]);
//                        printf("TESTE - %f\n",Teste.at<float>(contadorTeste,xx));
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

//PEGAR ANALISES//
void GetResults(float TP, float TN, float FP, float FN,float &speciticivy,float &sensitivity,float &ppv,float &accuracy){
    speciticivy+=(TN)/(FP+TN);
    sensitivity+=(TP)/(TP+FN);
    ppv+=(TP)/(TP+FP);
    accuracy=(TP+TN)/(TP+TN+FP+FN);
}
void GetTruesAndFalses(Mat Confusion, Mat ResPorClasse, int quantidadeClasses,int iteration,float &TP, float &TN, float &FP, float &FN){
    TP=Confusion.at<float>(iteration,iteration);
    for (int x=0; x<quantidadeClasses ; x++) {
        for (int y=0; y<quantidadeClasses; y++) {
            if (x==y)
                TN+=Confusion.at<float>(x,y);
            if (x==iteration)
                FP+=Confusion.at<float>(x,y);
            if (y==iteration) {
                FN+=Confusion.at<float>(x,y);
            }
        }
    }
    TN=TN-TP;
    FP=FP-TP;
    FN=FN-TP;
    float spe=0,sen=0,ppv=0,acc=0;
    GetResults(TP, TN, FP, FN, spe, sen, ppv, acc);
    ResPorClasse.at<float>(iteration,0)=spe;
    ResPorClasse.at<float>(iteration,1)=sen;
    ResPorClasse.at<float>(iteration,2)=acc;
    ResPorClasse.at<float>(iteration,3)=ppv;
}
void GetGeneralResults(Mat ResPorClasse,int quantidadeClasses,float &speciticivy,float &sensitivity,float &ppv,float &accuracy){
    speciticivy=0,sensitivity=0,ppv=0,accuracy=0;
    for (int x=0; x<4; x++) {
        float num=0;
        for (int y=0; y<quantidadeClasses; y++) {
            num+=ResPorClasse.at<float>(y,x);
        }
        num=num/quantidadeClasses;
        if (x==0)
            speciticivy=num;
        if (x==1)
            sensitivity=num;
        if (x==2)
            ppv=num;
        if (x==3)
            accuracy=num;
    }
}

//METODOS DE CLASSIFICACAO//
void Method_KNN(Mat Treino,Mat Teste,Mat TreinoLabel,Mat TesteLabel,Mat label,Mat atrib,float perTreino,float perTeste,int quantidadeClasses,int NumObjPorClasse[quantidadeClasses],float atributos,float objetos,int decision,Mat Confusion,int Media,Mat ResMedia){
    if (decision==1) {
        leave_on_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel,TesteLabel);
    }
    if (decision==2){
        hold_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel, TesteLabel);
    }
    if (Media==0) {
        rod++;
        K=K+2;
    }
    //------------------------------------------------------------
    //---------------------------KNN------------------------------
    //------------------------------------------------------------
    
    printf("Bando de dados: %s\n",address);
    printf("Metodo de Classificacao KNN %d\n",K);
    
    
    //------------------------------------------------------------
    //-----------------KNN TREINAMENTO----------------------------
    //------------------------------------------------------------
    
    
    Mat test_sample;
    KNearest knn(Treino,TreinoLabel,cv::Mat(), false, K);
    
    //------------------------------------------------------------
    //-----------------KNN CLASSIFICACAO--------------------------
    //------------------------------------------------------------
    
    float acerto=0;
    float erro=0;
    
    for (int tsample = 0; tsample < ((int)objetos*(perTeste/100)-1); tsample++) {
        
        test_sample=Teste.row(tsample);
        
        int res =knn.find_nearest(test_sample, K);
        int test = (int)(TesteLabel.at<float>(tsample));
        
        printf("KNN %d - Iteracao %d - Testing Sample %i -> class result (digit %d\t%d)\n",K,Media, tsample, res, test);
        
        if (test!=res){
            erro++;
            Confusion.at<float>(test,res)++;
        }
        else{
            acerto++;
            Confusion.at<float>(test,res)++;
        }
    }
    printf("Taxa de Acerto : %0.001f %%\n",((acerto*100)/(acerto+erro)));
    printf("\n_Fim_Do_KNN %d_\n",K);
    //GetResults
    Mat ResPorClasse(quantidadeClasses,4,CV_32FC1);
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<4; y++) {
            ResPorClasse.at<float>(x,y)=0;
        }
    }
    float TP=0,TN=0,FP=0,FN=0;
    float speciticivy=0,sensitivity=0,ppv=0,accuracy=0;
    for (int i=0; i<quantidadeClasses; i++){
        TP=0,TN=0,FP=0,FN=0;
        GetTruesAndFalses(Confusion,ResPorClasse,quantidadeClasses,i,TP,TN,FP,FN);
    }
    GetGeneralResults(ResPorClasse,quantidadeClasses,speciticivy,sensitivity,ppv,accuracy);
    ProgramToFile(Confusion, quantidadeClasses, atributos, objetos, acerto, erro,decision,perTreino,perTeste,speciticivy,sensitivity,ppv,accuracy,ResPorClasse,rod,ResMedia,Media);
}
void Method_Kmeans(Mat Treino,Mat Teste,Mat TreinoLabel,Mat TesteLabel,Mat label,Mat atrib,float perTreino,float perTeste,int quantidadeClasses,int NumObjPorClasse[quantidadeClasses],float atributos,float objetos, int decision,Mat Confusion,int Media,Mat ResMedia){
    if (decision==1) {
        leave_on_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel,TesteLabel);
    }
    if (decision==2){
        hold_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel, TesteLabel);
    }
    if (Media==0) {
        rod++;
    }
    float acerto=0,erro=0;
    
    //************KMEANS************//
    float* resultTest;
    resultTest=new float [quantidadeClasses];
    
    //Calculando a media na Mat Treino
    Mat TreinoKMeans(quantidadeClasses,atributos,CV_32FC1);
    for (int lbb=0; lbb<quantidadeClasses; lbb++) {
        for (int att=0; att<atributos; att++){
            float means=0;
            int cont=0;
            for (int objj=0; objj<int((objetos*perTreino/100)); objj++) {
                if (TreinoLabel.at<float>(objj,atributos)==lbb) {
                    means+=Treino.at<float>(objj,att);
                    cont++;
                }
            }
            means=means/cont;
            TreinoKMeans.at<float>(lbb,att)=means;
        }
    }
    //ExtraindoResultadosDeCadaClasse
    for (int matTest=0; matTest<(int(objetos*(perTeste/100))); matTest++){
        for (int matTreino=0; matTreino<quantidadeClasses; matTreino++) {
            float resultsLocal=0;
            for (int varrerAtributos=0; varrerAtributos<atributos; varrerAtributos++) {
                resultsLocal+=pow(Teste.at<float>(matTest,varrerAtributos)-TreinoKMeans.at<float>(matTreino,varrerAtributos),2);
            }
            resultTest[matTreino]=sqrt(resultsLocal);
        }
        //Achar o Menor e achar posicao do menor
        float me=100;
        int posMe=0;
        for (int menor=0; menor<quantidadeClasses; menor++) {
            if (me>resultTest[menor]) {
                me=resultTest[menor];
                posMe=menor;
            }
        }
        //Conferir
        if (posMe==TesteLabel.at<float>(matTest)) {
            acerto++;
            Confusion.at<float>(posMe,TesteLabel.at<float>(matTest))++;
        }
        else{
            erro++;
            Confusion.at<float>(posMe,TesteLabel.at<float>(matTest))++;
        }
        printf("Kmeans - Iteracao %d -Testing Sample %i -> class result (digit %d\t%1.f)\n",Media, matTest, posMe,TesteLabel.at<float>(matTest,1));
    }
    delete [] resultTest;
    printf("\nTaxa De Acerto - %0.001f%%\n",((acerto*100)/(acerto+erro)));
    printf("Acerto - %f\nErro - %f\n",acerto,erro);
    printf("\n_Fim_Do_KMEANS_\n");
    //GetResults
    Mat ResPorClasse(quantidadeClasses,4,CV_32FC1);
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<4; y++) {
            ResPorClasse.at<float>(x,y)=0;
        }
    }
    float TP=0,TN=0,FP=0,FN=0;
    float speciticivy=0,sensitivity=0,ppv=0,accuracy=0;
    for (int i=0; i<quantidadeClasses; i++){
        TP=0,TN=0,FP=0,FN=0;
        GetTruesAndFalses(Confusion,ResPorClasse,quantidadeClasses,i,TP,TN,FP,FN);
    }
    GetGeneralResults(ResPorClasse,quantidadeClasses,speciticivy,sensitivity,ppv,accuracy);
    ProgramToFile(Confusion, quantidadeClasses, atributos, objetos, acerto, erro,decision,perTreino,perTeste,speciticivy,sensitivity,ppv,accuracy,ResPorClasse,rod,ResMedia,Media);
}
void Method_SVM(Mat Treino,Mat Teste, Mat TreinoLabel, Mat TesteLabel,Mat label,Mat atrib, float perTreino, float perTeste,int quantidadeClasses,int NumObjPorClasse[quantidadeClasses],float atributos, float objetos,int decision,Mat Confusion,int kernell,int Media,Mat ResMedia){
    if (decision==1) {
        leave_on_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel,TesteLabel);
    }
    if (decision==2){
        hold_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel, TesteLabel);
    }
    if (Media==0) {
        rod++;
    }
    //SVM
    printf("Bando de dados: %s\n",address);
    printf("Metodo de Classificacao SVM\n");
    
    //------------------------------------------------------------
    //-----------------SVM PARAMETROS-----------------------------
    //------------------------------------------------------------
    
    //int kernell = 1; //Roda a SVM com kernell Linear
    //int kernell = 2; //Roda a SVM com kernell RBF
    //int kernell = 3; //Roda a SVM com kernell Poly
    //int kernell = 4; //Roda a SVM com kernell Sigmoid
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
        
        
        printf("SVM - Iteracao %d -Testing Sample %i -> class result (digit %d\t%d)\n",Media , tsample,res, test);
        
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
    if (kernell==1)
        printf("\n_Fim_Do_SVM_Linear_\n");
    if (kernell==2)
        printf("\n_Fim_Do_SVM_RBF_\n");
    if (kernell==3)
        printf("\n_Fim_Do_SVM_Poly_\n");
    if (kernell==4)
        printf("\n_Fim_Do_SVM_Sigmoid_\n");
    //GetResults
    Mat ResPorClasse(quantidadeClasses,4,CV_32FC1);
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<4; y++) {
            ResPorClasse.at<float>(x,y)=0;
        }
    }
    float TP=0,TN=0,FP=0,FN=0;
    float speciticivy=0,sensitivity=0,ppv=0,accuracy=0;
    for (int i=0; i<quantidadeClasses; i++){
        TP=0,TN=0,FP=0,FN=0;
        GetTruesAndFalses(Confusion,ResPorClasse,quantidadeClasses,i,TP,TN,FP,FN);
    }
    GetGeneralResults(ResPorClasse,quantidadeClasses,speciticivy,sensitivity,ppv,accuracy);
    ProgramToFile(Confusion, quantidadeClasses, atributos, objetos, acerto, erro,decision,perTreino,perTeste,speciticivy,sensitivity,ppv,accuracy,ResPorClasse,rod,ResMedia,Media);
}
void Method_Bayes(Mat Treino,Mat Teste,Mat TreinoLabel,Mat TesteLabel,Mat label,Mat atrib,float perTreino,float perTeste,int quantidadeClasses,int NumObjPorClasse[quantidadeClasses],float atributos,float objetos,int decision,Mat Confusion,int Media,Mat ResMedia){
    if (decision==1) {
        leave_on_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel,TesteLabel);
    }
    if (decision==2){
        hold_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel, TesteLabel);
    }
    if (Media==0) {
        rod++;
    }
    //------------------------------------------------------------
    //--------------------------Bayes-----------------------------
    //------------------------------------------------------------
    printf("Bando de dados: %s\n",address);
    printf("Metodo de Classificacao Bayes\n");
    
    //------------------------------------------------------------
    //-----------------Bayes TREINAMENTO--------------------------
    //------------------------------------------------------------
    
    
    CvNormalBayesClassifier *bayes = new CvNormalBayesClassifier();
    
    printf("\nTreinando Bayes...");
    bayes->train(Treino, TreinoLabel);
    printf("OK!\n\n");
    
    //------------------------------------------------------------
    //-------------------Bayes CLASSIFICACAO----------------------
    //------------------------------------------------------------
    
    Mat test_sample;
    
    float acerto=0,erro=0;
    
    for (int tsample = 0; tsample < ((int)(objetos*(perTeste/100))-1); tsample++) {
        
        
        test_sample = Teste.row(tsample);
        
        int res = (int) (bayes->predict(test_sample));
        
        int test = (int) (TesteLabel.at<float>(tsample));
        
        printf("Bayes - Iteracao %d -Testing Sample %i -> class result (digit %d\t%d)\n",Media, tsample, res, test);
        
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
    printf("\n_Fim_Do_Bayes_\n");
    //GetResults
    Mat ResPorClasse(quantidadeClasses,4,CV_32FC1);
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<4; y++) {
            ResPorClasse.at<float>(x,y)=0;
        }
    }
    float TP=0,TN=0,FP=0,FN=0;
    float speciticivy=0,sensitivity=0,ppv=0,accuracy=0;
    for (int i=0; i<quantidadeClasses; i++){
        TP=0,TN=0,FP=0,FN=0;
        GetTruesAndFalses(Confusion,ResPorClasse,quantidadeClasses,i,TP,TN,FP,FN);
    }
    GetGeneralResults(ResPorClasse,quantidadeClasses,speciticivy,sensitivity,ppv,accuracy);
    ProgramToFile(Confusion, quantidadeClasses, atributos, objetos, acerto, erro,decision,perTreino,perTeste,speciticivy,sensitivity,ppv,accuracy,ResPorClasse,rod,ResMedia,Media);
}
void Method_MLP(Mat Treino,Mat Teste,Mat TreinoLabel,Mat TesteLabel,Mat label,Mat atrib,float perTreino,float perTeste,int quantidadeClasses,int NumObjPorClasse[quantidadeClasses],float atributos,float objetos,int decision,Mat Confusion,int config,int Media,Mat ResMedia){
    if (decision==1) {
        leave_on_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel,TesteLabel);
    }
    if (decision==2){
        hold_out(perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, label, atrib, Treino, Teste, TreinoLabel, TesteLabel);
    }
    if (Media==0) {
        rod++;
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
    
    printf("\nBando de dados: %s\n",address);
    printf("Metodo de Classificacao MLP\n");
    
    cv::Mat classificationResult(1, quantidadeClasses, CV_32FC1);
    
    
    //------------------------------------------------------------
    //-----------------MLP TREINAMENTO----------------------------
    //------------------------------------------------------------
    
    
    cv::Mat layers(3,1,CV_32S);
    
    int numHiddenLayer;
    //int config = 1; //Roda a configuracao 1 da MLP
    //int config = 2; //Roda a configuracao 2 da MLP
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
    printf("Iteracao %d\n",Media);
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
        
        printf("MLP Config %d - Testing Sample %i -> class result (digit %d)\n",config, tsample, maxIndex);
        
        if (teste_lable_MLP.at<float>(tsample, maxIndex)!=1.0f){
            erro++;
            for(int class_index=0;class_index<quantidadeClasses;class_index++){
                if(teste_lable_MLP.at<float>(tsample, class_index)==1.0f){
                    Confusion.at<float>(maxIndex,class_index)++;
                    break;
                }
            }
        }
        else{
            Confusion.at<float>(maxIndex,maxIndex)++;
            acerto++;
        }
    }
    printf("Taxa de Acerto : %0.001f %%",((acerto*100)/(acerto+erro)));
    printf("\n_Fim_Do_MLP_Config0%d_\n",config);
    //GetResults
    Mat ResPorClasse(quantidadeClasses,4,CV_32FC1);
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<4; y++) {
            ResPorClasse.at<float>(x,y)=0;
        }
    }
    float TP=0,TN=0,FP=0,FN=0;
    float speciticivy=0,sensitivity=0,ppv=0,accuracy=0;
    for (int i=0; i<quantidadeClasses; i++){
        TP=0,TN=0,FP=0,FN=0;
        GetTruesAndFalses(Confusion,ResPorClasse,quantidadeClasses,i,TP,TN,FP,FN);
    }
    GetGeneralResults(ResPorClasse,quantidadeClasses,speciticivy,sensitivity,ppv,accuracy);
    ProgramToFile(Confusion, quantidadeClasses, atributos, objetos, acerto, erro,decision,perTreino,perTeste,speciticivy,sensitivity,ppv,accuracy,ResPorClasse,rod,ResMedia,Media);
}

//MAIN//
int main(){
    int decision;
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
    
    //Aplicacao do Metodo de Separacao
    printf("Selecione:\n1-Leave_On_Out\n2-Hold-out\n");
    scanf("%d",&decision);
    
    //Criar Mat Confusion
    Mat Confusion(quantidadeClasses,quantidadeClasses,CV_32FC1);
    for (int x=0; x<quantidadeClasses; x++) {
        for (int y=0; y<quantidadeClasses; y++) {
            Confusion.at<float>(x,y)=0;
        }
    }
    //Declaracao da Mat ResMedia
    Mat ResMedia (10,4,CV_32FC1);
    InitializeArquivo();
    //Aplicacao do Metodo KNN(1,3,5)
    for (int p=0; p<3; p++) {
        for (int Media=0; Media<10; Media++) {
            Method_KNN(Treino, Teste, TreinoLabel, TesteLabel, label, atrib, perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, decision, Confusion,Media,ResMedia);
        }
    }
    //Aplicacao do Metodo KMEANS
    for (int Media=0; Media<10; Media++) {
        Method_Kmeans(Treino, Teste, TreinoLabel, TesteLabel, label, atrib, perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, decision, Confusion,Media,ResMedia);
    }
    //Aplicacao do Metodo SVM(Linear,RBF,Poly,Sigmoid)
    int kernell;
    for (int p=1; p<5; p++) {
        kernell=p;
        for (int Media=0; Media<10; Media++){
            Method_SVM(Treino, Teste, TreinoLabel, TesteLabel, label, atrib, perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, decision, Confusion, kernell,Media,ResMedia);
        }
    }
    //Aplicacao do Metodo Bayes
    for (int Media=0; Media<10; Media++){
        Method_Bayes(Treino, Teste, TreinoLabel, TesteLabel, label, atrib, perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, decision, Confusion,Media,ResMedia);
    }
    //Aplicacao do Metodo MLP
    int config;
    for (int p=1; p<3; p++) {
        config=p;
        for (int Media=0; Media<10; Media++){
            Method_MLP(Treino, Teste, TreinoLabel, TesteLabel, label, atrib, perTreino, perTeste, quantidadeClasses, NumObjPorClasse, atributos, objetos, decision, Confusion, config,Media,ResMedia);
        }
    }
    printf("\nFIM_DO_PROGRAMA\n");
}