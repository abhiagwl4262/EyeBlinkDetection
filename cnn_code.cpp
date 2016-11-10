#include <opencv2/opencv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv/cv.hpp>
#include <opencv/highgui.h>
//#include <math.h>
#include <fstream>
#include <iostream>
//#include <time.h>
#include <stdio.h>
#include <random>

using namespace cv;
using namespace std;

//#define Leaky_relu
//#define SHOW_TEST_IMAGE
#define Dropout_layer
//#define test_on_train_data
//#define read_init
#define MAX_ITERATION 0
//#define SKIP_TRAIN
#define INTERMEDIATE_TEST
//#define NO_BATCH
//#define WRITE_TEST_IMAGE

int NUMBER_TRAIN_IMAGES;
int NUMBER_TEST_IMAGES;

#define G_CHECKING 0

// Conv2 parameter
#define CONV_FULL 0
#define CONV_SAME 1
#define CONV_VALID 2

// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1
#define POOL_STOCHASTIC 2

#define loop_count 1000
#define drop 0.5
#define ATD at<double>
#define elif else if
int NumHiddenNeurons[2] = {400,300};
int NumHiddenLayers = 2;
int nclasses = 4;
int KernelSize = 13;
int KernelAmount = 8;
int PoolingDim = 4;
int Pooling_Method = POOL_MAX;

int batch;
Mat Dropout[2];
Mat Probability[2];
FILE* logfile;

void readWeight(Mat M, string s);
void saveWeight(Mat &M, string s);

// Bernoulli_Distribution (NOT USEFUL RIGHT NOW)
Mat Bernauli2(int neurons, float p){
    Mat temp = Mat::zeros(neurons,1, CV_64FC1);
    int MaxCOUNT_1 = neurons*p;
    for(int i = 0; i<MaxCOUNT_1; i++){
        int v1 = rand() % neurons;
        if (temp.ATD(v1,0)!= 1)
        {
            temp.ATD(0,v1) = 1;
        }
        else
        {
          i--;
          continue;
        }
    }
    return temp;
}
// Bernoulli_Distribution (NOT USEFUL RIGHT NOW)
Mat Bernauli1(int neurons, float p){
    int MaxCOUNT_1 = neurons*p;
    int MaxCOUNT_0 = neurons*(1-p);
    int count1, count0 = 0;
    Mat M = Mat:: zeros(neurons,1,CV_64FC1);
    int j =0;
    while(count1!=MaxCOUNT_1 || count0!=MaxCOUNT_0){
        int v1 = rand() % 2;
        M.ATD(j,0) = v1;
        j++;
        if (v1== 1) count1++;
        else count0++;
    
    if (count1==MaxCOUNT_1){
        for (int i =j ; j<neurons ;j++){
            M.ATD(j,0) = 0;
        }
    }
    else{
        for (int i =j ; j<neurons ;j++){
            M.ATD(j,0) = 1;
        }
    }
    }
    return M ;
}

// Bernoulli_Distribution ()
Mat 
Bernauli(int neurons, float p){
  int count=0;  // count number of trues
  Mat M = Mat:: zeros(neurons,1,CV_64FC1);
  randu(M , Scalar(1.0), Scalar(0.0));
  for(int j=0; j<neurons; j++){
      if (M.ATD(j,0)> p) 
      {
          M.ATD(j,0) = 1.0;
      }
      else 
      {
          M.ATD(j,0) = 0.0;
      }
   }
  //saveWeight(M,"prob");
  //cout << count << endl;
  //waitKey(0);
  return M;
}

typedef struct ConvKernel{
    Mat W;
    double b;
    Mat Wgrad;
    double bgrad;
}ConvK;

typedef struct ConvLayer{
    vector<ConvK> layer;
    int kernelAmount;
}Cvl;

typedef struct Network{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
}Ntw;

typedef struct SoftmaxRegession{
    Mat Weight;
    Mat Wgrad;
    Mat b;
    Mat bgrad;
    double cost;
}SMR;

Mat resultPredict(vector<Mat> &x, Mat &y, Cvl &cvl, vector<Ntw> &hLayers, SMR &smr, double lambda, Mat neuron_prob[]);

Mat concatenateMat(vector<vector<Mat> > &vec){

    int subFeatures = vec[0][0].rows * vec[0][0].cols;
    int height = vec[0].size() * subFeatures;
    int width = vec.size();
    Mat res = Mat::zeros(height, width, CV_64FC1);

    for(int i=0; i<vec.size(); i++){
        for(int j=0; j<vec[i].size(); j++){
            Rect roi = Rect(i, j * subFeatures, 1, subFeatures);                                                                                                                                                          
            Mat subView = res(roi);
            Mat ptmat = vec[i][j].reshape(0, subFeatures);
            ptmat.copyTo(subView);
        }
    }
    return res;
}

Mat concatenateMat(vector<Mat> &vec){

    int height = vec[0].rows;
    int width = vec[0].cols;
    Mat res = Mat::zeros(height * width, vec.size(), CV_64FC1);
    for(int i=0; i<vec.size(); i++){
        Mat img(vec[i]);
        // reshape(int cn, int rows=0), cn is num of channels.
        Mat ptmat = img.reshape(0, height * width);
        Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
        Mat subView = res(roi);
        ptmat.copyTo(subView);
    }
    return res;
}

void
unconcatenateMat(Mat &M, vector<vector<Mat> > &vec, int vsize){

    int sqDim = M.rows / vsize;
    int Dim = sqrt((double)sqDim);
    for(int i=0; i<M.cols; i++){
        vector<Mat> oneColumn;
        for(int j=0; j<vsize; j++){
            Rect roi = Rect(i, j * sqDim, 1, sqDim);
            Mat temp;
            M(roi).copyTo(temp);
            Mat img = temp.reshape(0, Dim);
            oneColumn.push_back(img);
        }
        vec.push_back(oneColumn);
    }
}

Mat 
sigmoid(Mat &M){
    Mat temp;
    exp(-M, temp);
    return 1.0 / (temp + 1.0);
}

Mat 
dsigmoid(Mat &a){
    Mat res = 1.0 - a;
    res = res.mul(a);
    return res;
}

Mat
ReLU(Mat& M){
    Mat res(M);
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            if(M.ATD(i, j) < 0.0){
                res.ATD(i, j) = 0.0;

            #ifdef Leaky_relu
                res.ATD(i, j) = res.ATD(i, j)*0.005 ; 
            #endif
            }
        }
    }
    return res;
}

Mat
dReLU(Mat& M){
    Mat res = Mat::zeros(M.rows, M.cols, CV_64FC1);
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            if(M.ATD(i, j) > 0.0) res.ATD(i, j) = 1.0;
        }
    }
    return res;
}

// Mimic rot90() in Matlab/GNU Octave.
Mat 
rot90(Mat &M, int k){
    Mat res;
    if(k == 0) return M;
    elif(k == 1){
        flip(M.t(), res, 0);
    }else{
        flip(rot90(M, k - 1).t(), res, 0);
    }
    return res;
}


// A Matlab/Octave style 2-d convolution function.

Mat 
conv2(Mat &img, Mat &kernel, int convtype) {
    Mat dest;
    Mat source = img;
    if(CONV_FULL == convtype) {
        source = Mat();
        int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
        copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
    }
    Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
    int borderMode = BORDER_CONSTANT;
    Mat fkernal;
    flip(kernel, fkernal, -1);
    filter2D(source, dest, img.depth(), fkernal, anchor, 0, borderMode);

    if(CONV_VALID == convtype) {
        dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
                   .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
    }
    return dest;
}

// get KroneckerProduct for upsample
// see function kron() in Matlab/Octave
Mat
kron(Mat &a, Mat &b){

    Mat res = Mat::zeros(a.rows * b.rows, a.cols * b.cols, CV_64FC1);
    for(int i=0; i<a.rows; i++){
        for(int j=0; j<a.cols; j++){
            Rect roi = Rect(j * b.cols, i * b.rows, b.cols, b.rows);
            Mat temp = res(roi);
            Mat c = b.mul(a.ATD(i, j));
            c.copyTo(temp);
        }
    }
    return res;
}

Point
findLoc(double val, Mat &M){
    Point res = Point(0, 0);
    double minDiff = 1e8;
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            if(val >= M.ATD(i, j) && (val - M.ATD(i, j) < minDiff)){
                minDiff = val - M.ATD(i, j);
                res.x = j;
                res.y = i;
            }
        }
    }
    return res;
}

Mat
Pooling(Mat &M, int pVert, int pHori, int poolingMethod, vector<Point> &locat, bool isTest){
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX, remY, M.cols - remX, M.rows - remY);
        M(roi).copyTo(newM);
    }
    Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC1);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);
            newM(roi).copyTo(temp);
            double val;
            // for Max Pooling
            if(POOL_MAX == poolingMethod){ 
                double minVal; 
                double maxVal; 
                Point minLoc; 
                Point maxLoc;
                minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc );
                val = maxVal;
                locat.push_back(Point(maxLoc.x + j * pHori, maxLoc.y + i * pVert));
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = sum(temp)[0] / (pVert * pHori);
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                double sumval = sum(temp)[0];
                Mat prob = temp / sumval;
                if(isTest){
                    val = sum(prob.mul(temp))[0];
                }else{
                    RNG rng;
                    double ran = rng.uniform((double)0, (double)1);
                    double minVal; 
                    double maxVal; 
                    Point minLoc; 
                    Point maxLoc;
                    minMaxLoc( prob, &minVal, &maxVal, &minLoc, &maxLoc );
                    ran *= maxVal;
                    Point loc = findLoc(ran, prob);
                    val = temp.ATD(loc.y, loc.x);
                    locat.push_back(Point(loc.x + j * pHori, loc.y + i * pVert));
                }

            }
            res.ATD(i, j) = val;
        }
    }
    return res;
}

Mat 
UnPooling(Mat &M, int pVert, int pHori, int poolingMethod, vector<Point> &locat){
    Mat res;
    if(POOL_MEAN == poolingMethod){
        Mat one = Mat::ones(pVert, pHori, CV_64FC1);
        res = kron(M, one) / (pVert * pHori);
    }elif(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod){
        res = Mat::zeros(M.rows * pVert, M.cols * pHori, CV_64FC1);
        for(int i=0; i<M.rows; i++){
            for(int j=0; j<M.cols; j++){
                res.ATD(locat[i * M.cols + j].y, locat[i * M.cols + j].x) = M.ATD(i, j);
            }
        }
    }
    return res;
}

double 
getLearningRate(Mat &data){
    // see Yann LeCun's Efficient BackProp, 5.1 A Little Theory
    int nfeatures = data.rows;
    int nsamples = data.cols;
    //covariance matrix = x * x' ./ nfeatures;
    Mat Sigma = data * data.t() / nsamples;
    SVD uwvT = SVD(Sigma);
    return 0.9 / uwvT.w.ATD(0, 0);
}

void
weightRandomInit(ConvK &convk, int width, int j){

    double epsilon = 0.1;
    convk.W = Mat::ones(width, width, CV_64FC1);
    
    #ifdef SKIP_TRAIN

    // Read Weight from text files
    char file[50];
    sprintf(file,"w%d_7",j);
    readWeight(convk.W, file);

    //read bias
    sprintf(file,"b%d_7.txt",j);
    std::fstream my(file,ios::in);
    if(my.is_open())
    {
        my>>convk.b;
    }

    #else

      #ifdef read_init
        char file[100];
        sprintf(file,"W%d_74",j);
        readWeight(convk.W, file);
        //read bias
        sprintf(file,"b%d_74.txt",j);
        std::fstream my(file,ios::in);
        if(my.is_open())
        {
            my>>convk.b;
        }
      #else
        double *pData; 
        for(int i = 0; i<convk.W.rows; i++){
            pData = convk.W.ptr<double>(i);
            for(int j=0; j<convk.W.cols; j++){
                pData[j] = (rand()%1000)/(double)1000; 
            }
        }
            convk.W = convk.W * (2 * epsilon) - epsilon;
            convk.b = 0;
      #endif
    #endif
    
    convk.Wgrad = Mat::zeros(width, width, CV_64FC1);
    convk.bgrad = 0;
}

void
weightRandomInit(Ntw &ntw, int inputsize, int hiddensize, int nsamples, int j){

    double epsilon = sqrt((double)6) / sqrt((double)hiddensize + inputsize + 1);
    double *pData;
    ntw.W = Mat::ones(hiddensize, inputsize, CV_64FC1);
    ntw.b = Mat::zeros(hiddensize, 1, CV_64FC1);

#ifdef SKIP_TRAIN
    // Read Weight from text files
    char file[50];
    sprintf(file, "HL%d_7", j);
    readWeight(ntw.W, file);

    // Read bias from text files
    sprintf(file, "HL_b%d_7", j);
    readWeight(ntw.b, file);

#else

 #ifdef read_init
    char file[100];
    sprintf(file,"HL%d_74",j);
    readWeight(ntw.W, file);
    // Read bias from text files
    sprintf(file, "HL_b%d_74", j);
    readWeight(ntw.b, file);
 #else
    for(int i=0; i<hiddensize; i++){
        pData = ntw.W.ptr<double>(i);
        for(int j=0; j<inputsize; j++){
            pData[j] = (rand() % 1000) / (double)1000;
        }
    }
    ntw.W = ntw.W * (2 * epsilon) - epsilon;
    ntw.b = Mat::zeros(hiddensize, 1, CV_64FC1);
 #endif

#endif
    ntw.Wgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.bgrad = Mat::zeros(hiddensize, 1, CV_64FC1);
}
void 
weightRandomInit(SMR &smr, int nclasses, int nfeatures){
    double epsilon = 0.01;
    smr.Weight = Mat::ones(nclasses, nfeatures, CV_64FC1);
    smr.b = Mat::zeros(nclasses, 1, CV_64FC1);
#ifdef SKIP_TRAIN
    // Read Weight from text files
    readWeight(smr.Weight, "smr_7");

    // Read bias from text files
    readWeight(smr.b, "smr_b_7");
#else
  #ifdef read_init
    char file[100];
    sprintf(file,"smr_74");
    readWeight(smr.Weight, file);
    // Read bias from text files
    readWeight(smr.b, "smr_b_74");
  #else
    double *pData; 
    for(int i = 0; i<smr.Weight.rows; i++){
        pData = smr.Weight.ptr<double>(i);
        for(int j=0; j<smr.Weight.cols; j++){
            // pData[j] = randu<double>();        
            pData[j] = (rand() % 1000) / (double)1000;
        }
    }
    smr.Weight = smr.Weight * (2 * epsilon) - epsilon;
    smr.b = Mat::zeros(nclasses, 1, CV_64FC1);
  #endif

#endif
    smr.cost = 0.0;
    smr.Wgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.bgrad = Mat::zeros(nclasses, 1, CV_64FC1);
}

void
ConvNetInitPrarms(Cvl &cvl, vector<Ntw> &HiddenLayers, SMR &smr, int imgDim, int nsamples){

    // Init Conv layers
    
    for(int j=0; j<KernelAmount; j++){
        ConvK tmpConvK;
        weightRandomInit(tmpConvK, KernelSize, j);
        cvl.layer.push_back(tmpConvK);

    #ifndef SKIP_TRAIN
        char file[20];
        sprintf(file, "W%d_init", j);
        saveWeight(tmpConvK.W, file);
    #endif
    }

    // Init Hidden layers
    cvl.kernelAmount = KernelAmount;
    int outDim = imgDim - KernelSize + 1; 
    outDim = outDim / PoolingDim;
    int hiddenfeatures = pow((double)outDim, 2) * KernelAmount;
    Ntw tpntw;

    weightRandomInit(tpntw, hiddenfeatures, NumHiddenNeurons[0], nsamples,1);
    HiddenLayers.push_back(tpntw);

#ifndef SKIP_TRAIN
    char file[20];
    sprintf(file, "HL1_init");
    saveWeight(tpntw.W, file);
#endif
    for(int i=1; i<NumHiddenLayers; i++){
        Ntw tpntw2;
        weightRandomInit(tpntw2, NumHiddenNeurons[0], NumHiddenNeurons[1], nsamples,i+1);
        HiddenLayers.push_back(tpntw2);
        #ifndef SKIP_TRAIN
        char file[20];
        sprintf(file, "HL%d_init", i+1);
        saveWeight(tpntw2.W, file);
        #endif
    }

    // Init Softmax layer
    weightRandomInit(smr, nclasses, NumHiddenNeurons[1]);
#ifndef SKIP_TRAIN
    sprintf(file, "smr_init");
    saveWeight(smr.Weight, file);
#endif
}

Mat
getNetworkActivation(Ntw &ntw, Mat &data){
    Mat acti;
    acti = ntw.W * data + repeat(ntw.b, 1, data.cols);
    acti = sigmoid(acti);
    return acti;
}


void
getNetworkCost(vector<Mat> &x, Mat &y, Cvl &cvl, vector<Ntw> &hLayers, SMR &smr, double lambda){

    int nsamples = x.size();
    // Conv & Pooling
    vector<vector<Mat> > Conv1st;
    vector<vector<Mat> > Pool1st;
    vector<vector<vector<Point> > > PoolLoc;
    for(int k=0; k<nsamples; k++){
        vector<Mat> tpConv1st;
        vector<Mat> tpPool1st;
        vector<vector<Point> > PLperSample;
        for(int i=0; i<cvl.kernelAmount; i++){ 
            vector<Point> PLperKernel;
            Mat temp = rot90(cvl.layer[i].W, 2);
            Mat tmpconv = conv2(x[k], temp, CONV_VALID);
            tmpconv += cvl.layer[i].b;
            //tmpconv = sigmoid(tmpconv);
            tmpconv = ReLU(tmpconv);
            tpConv1st.push_back(tmpconv);
            tmpconv = Pooling(tmpconv, PoolingDim, PoolingDim, Pooling_Method, PLperKernel, false);
            PLperSample.push_back(PLperKernel);
            tpPool1st.push_back(tmpconv);
        }
        PoolLoc.push_back(PLperSample);
        Conv1st.push_back(tpConv1st);
        Pool1st.push_back(tpPool1st);
    }
    Mat convolvedX = concatenateMat(Pool1st);

    // full connected layers

    vector<Mat> acti;
    acti.push_back(convolvedX);

    #ifdef Dropout_layer
    Mat temp_Dropout[2];
    for(int i = 0; i < NumHiddenLayers; i++)
    {
        Dropout[i] = Bernauli(NumHiddenNeurons[i],drop);
        temp_Dropout[i] = repeat(Dropout[i], 1, nsamples);
    }
    #endif
    
    for(int i=1; i<=NumHiddenLayers; i++){
        Mat tmpacti = hLayers[i-1].W * acti[i-1] + repeat(hLayers[i-1].b, 1, convolvedX.cols);
        #ifdef Dropout_layer
            acti.push_back(sigmoid(tmpacti).mul(temp_Dropout[i-1]));
        #else
            acti.push_back(sigmoid(tmpacti));
        #endif
    }

    Mat M = smr.Weight * acti[acti.size() - 1] + repeat(smr.b, 1, nsamples);
    Mat tmp;
    reduce(M, tmp, 0, CV_REDUCE_MAX);
    M -= repeat(tmp, M.rows, 1);
    Mat p;
    exp(M, p);
    reduce(p, tmp, 0, CV_REDUCE_SUM);
    divide(p, repeat(tmp, p.rows, 1), p);

    // softmax regression
    Mat groundTruth = Mat::zeros(nclasses, nsamples, CV_64FC1);
    for(int i=0; i<nsamples; i++){
        groundTruth.ATD(y.ATD(0, i), i) = 1.0;
    }
    
    Mat logP;
    log(p, logP);
    logP = logP.mul(groundTruth);
    smr.cost = - sum(logP)[0] / nsamples;
    pow(smr.Weight, 2.0, tmp);
    smr.cost += sum(tmp)[0] * lambda / 2;
    for(int i=0; i<cvl.kernelAmount; i++){
        pow(cvl.layer[i].W, 2.0, tmp);
        smr.cost += sum(tmp)[0] * lambda / 2;
    }

    // bp - softmax
    tmp = (groundTruth - p) * acti[acti.size() - 1].t();
    tmp /= -nsamples;
    smr.Wgrad = tmp + lambda * smr.Weight;
    reduce((groundTruth - p), tmp, 1, CV_REDUCE_SUM);
    smr.bgrad = tmp / -nsamples;

    // bp - full connected
    vector<Mat> delta(acti.size());
    delta[delta.size() -1] = -smr.Weight.t() * (groundTruth - p);
    delta[delta.size() -1] = delta[delta.size() -1].mul(dsigmoid(acti[acti.size() - 1]));
    for(int i = delta.size() - 2; i >= 0; i--){
        delta[i] = hLayers[i].W.t() * delta[i + 1];
        if(i > 0) delta[i] = delta[i].mul(dsigmoid(acti[i]));
    }

    Mat tmp_Dropout[2];
    for(int i = 0; i < NumHiddenLayers; i++)
    {
        tmp_Dropout[i] = repeat(Dropout[i], 1, NumHiddenNeurons[i]);
    }
    for(int i=NumHiddenLayers - 1; i >=0; i--){

    #ifdef Dropout_layer
        hLayers[i].Wgrad = delta[i + 1].mul(temp_Dropout[i])*acti[i].t();  
        hLayers[i].Wgrad /= nsamples;
        reduce(delta[i + 1], tmp, 1, CV_REDUCE_SUM);
        hLayers[i].bgrad = tmp.mul(Dropout[i]) / nsamples;   

    #else
        hLayers[i].Wgrad = delta[i + 1]*acti[i].t();  
        hLayers[i].Wgrad /= nsamples;
        reduce(delta[i + 1], tmp, 1, CV_REDUCE_SUM);
        hLayers[i].bgrad = tmp / nsamples;    
    #endif
    }
    //bp - Conv layer
    Mat one = Mat::ones(PoolingDim, PoolingDim, CV_64FC1);
    vector<vector<Mat> > Delta;
    vector<vector<Mat> > convDelta;
    unconcatenateMat(delta[0], Delta, cvl.kernelAmount);
    for(int k=0; k<Delta.size(); k++){
        vector<Mat> tmp;
        for(int i=0; i<Delta[k].size(); i++){
            Mat upDelta = UnPooling(Delta[k][i], PoolingDim, PoolingDim, Pooling_Method, PoolLoc[k][i]);
            //upDelta = upDelta.mul(dsigmoid(Conv1st[k][i]));
            upDelta = upDelta.mul(dReLU(Conv1st[k][i]));
            tmp.push_back(upDelta);
        }
        convDelta.push_back(tmp); 
    }
    
    for(int j=0; j<cvl.kernelAmount; j++){
        Mat tpgradW = Mat::zeros(KernelSize, KernelSize, CV_64FC1);
        double tpgradb = 0.0;
        for(int i=0; i<nsamples; i++){
            Mat temp = rot90(convDelta[i][j], 2);
            tpgradW += conv2(x[i], temp, CONV_VALID);
            tpgradb += sum(convDelta[i][j])[0];
        }
        cvl.layer[j].Wgrad = tpgradW / nsamples + lambda * cvl.layer[j].W;
        cvl.layer[j].bgrad = tpgradb / nsamples;
    }
    // deconstruct
    int o =Conv1st.size();
    for(int i=0; i<o; i++){
        Conv1st[i].clear();
        Pool1st[i].clear();
    }
    Conv1st.clear();
    Pool1st.clear();
    int n = PoolLoc.size();
    for(int i=0; i<n; i++){
        int m = PoolLoc[i].size();
        for(int j=0; j<m; j++){
            PoolLoc[i][j].clear();
        }
        PoolLoc[i].clear();
    }
    PoolLoc.clear();
    acti.clear();
    delta.clear();
}

void
gradientChecking(Cvl &cvl, vector<Ntw> &hLayers, SMR &smr, vector<Mat> &x, Mat &y, double lambda){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(x, y, cvl, hLayers, smr, lambda);
    Mat grad(cvl.layer[0].Wgrad);
    cout<<"test network !!!!"<<endl;
    double epsilon = 1e-4;
    for(int i=0; i<cvl.layer[0].W.rows; i++){
        for(int j=0; j<cvl.layer[0].W.cols; j++){
            double memo = cvl.layer[0].W.ATD(i, j);
            cvl.layer[0].W.ATD(i, j) = memo + epsilon;
            getNetworkCost(x, y, cvl, hLayers, smr, lambda);
            double value1 = smr.cost;
            cvl.layer[0].W.ATD(i, j) = memo - epsilon;
            getNetworkCost(x, y, cvl, hLayers, smr, lambda);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<grad.ATD(i, j) / tp<<endl;
            cvl.layer[0].W.ATD(i, j) = memo;
        }
    }
}

void trainNetwork(vector<Mat> &x, Mat &y, vector<Mat> &x1, Mat &y1, Cvl &cvl, vector<Ntw> &HiddenLayers, SMR &smr, double lambda, int MaxIter, double lrate)
{

   //  if (G_CHECKING){
   //     gradientChecking(cvl, HiddenLayers, smr, x, y, lambda);
  //  }else{
        int converge = 0;
        double lastcost = 0.0;
        Mat last[2], Probability[2];
        for(int i = 0; i < NumHiddenLayers; i++)
        {
            last[i] = Mat:: zeros(NumHiddenNeurons[i],1,CV_64FC1);
            Probability[i] = Mat:: zeros(NumHiddenNeurons[i],1,CV_64FC1);
        }
    
        cout<<"Network Learning, trained learning rate: "<<lrate<<endl;
    
    double end_cost,start_cost;
    start_cost = clock();

#if (MAX_ITERATION != 0)
        while(converge < MaxIter)
#else
        while(1)
#endif
        {
#ifndef NO_BATCH
            int randomNum =((long)rand() + (long)rand()) % (x.size() - batch);
#endif
           vector<Mat> batchX;
           for(int i=0; i<batch; i++){
            #ifdef NO_BATCH
                    batchX.push_back(x[i]);
            #else
                    batchX.push_back(x[i + randomNum]);
            #endif
            }
            #ifndef NO_BATCH 
                    Rect roi = Rect(randomNum, 0, batch, y.rows);
            #else
                    Rect roi = Rect(0, 0, batch, y.rows);
            #endif

            Mat batchY = y(roi);
            
           
            getNetworkCost(batchX, batchY, cvl, HiddenLayers, smr, lambda);
            

            #ifndef NO_BATCH
                    cout<<"learning step: "<<converge<<", Cost function value = "<<smr.cost<<", randomNum = "<<randomNum<<endl;
            #else
                    cout<<"learning step: "<<converge<<", Cost function value = "<<smr.cost<<endl;
            #endif
            
            if(fabs((smr.cost - lastcost) / smr.cost) <= 1e-7 && converge > 0) break;
            if(smr.cost <= 0) break;
            lastcost = smr.cost;
            smr.Weight -= lrate * smr.Wgrad;
            smr.b -= lrate * smr.bgrad;

            #ifdef Dropout_layer
            for(int i = 0; i < NumHiddenLayers; i++)
            {   
                last[i] += Dropout[i];
                Probability[i] = 1.0*last[i]/(converge+1);
                 //cout<<Probability<< endl;
                char file[100];
                sprintf(file,"neuron_probability_%d",i);
                saveWeight(Probability[i], file);
            }
            #endif

            for(int i=0; i<HiddenLayers.size(); i++){
                //cout<<HiddenLayers[1].Wgrad<<" ";
                HiddenLayers[i].W -= lrate * HiddenLayers[i].Wgrad;
                HiddenLayers[i].b -= lrate * HiddenLayers[i].bgrad;
            }
            for(int i=0; i<cvl.kernelAmount; i++){
                cvl.layer[i].W -= lrate * cvl.layer[i].Wgrad;
                cvl.layer[i].b -= lrate * cvl.layer[i].bgrad;
            }
            ++ converge;
            
#ifdef INTERMEDIATE_TEST
    if(converge % loop_count == 0){

        for(int i = 0 ; i<KernelAmount ;i++){
            char file[20];
            sprintf(file,"b%d_%d.txt",i, (int)(converge/loop_count));
            FILE *File  = fopen( file ,"w");
            fprintf(File , "%.12f" , cvl.layer[i].b);
            fclose(File);
        }
        for(int i = 0 ; i<KernelAmount ;i++){
            char file[20];
            sprintf(file,"w%d_%d",i, (int)(converge/loop_count));
            saveWeight(cvl.layer[i].W, file);
        }
        
        for(int i = 0 ; i<NumHiddenLayers ;i++){
            char file[20];
            sprintf(file,"HL_b%d_%d",i+1, (int)(converge/loop_count));
            saveWeight(HiddenLayers[i].b, file);

            sprintf(file,"HL%d_%d",i+1, (int)(converge/loop_count));
            saveWeight(HiddenLayers[i].W, file);
        }
        char file[20];
        sprintf(file,"smr_b_%d",(int)(converge/loop_count));
        saveWeight(smr.b, file);
        sprintf(file,"smr_%d",(int)(converge/loop_count));
        saveWeight(smr.Weight, file);

    
        /*Checking test data accuracy*/

        //startt = clock();
        logfile = fopen("logfile.txt","a+");
        Mat result = resultPredict(x1, y1, cvl, HiddenLayers, smr, 3e-3, Probability);
        for (int i = 0; i < result.cols; i++)
        {
        printf("class :: %f\tcorrect class::%f\n", result.at<double>(0, i),y1.at<double>(0, i));
        fprintf(logfile,"class :: %f\tcorrect class::%f\n", result.at<double>(0, i),y1.at<double>(0, i));
        }
        
        Mat err;
        y1.copyTo(err);
        err -= result;
        int correct = err.cols;
        for(int i=0; i<err.cols; i++){
            if(err.ATD(0, i) != 0)
            {
                --correct;
            }
        }
        cout << "test data cost value ::" << smr.cost << endl ;     
        cout<<"test data :: correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
        fprintf(logfile,"correct: %d, accuracy: %f\n", err.cols,((double)correct / (double)err.cols));
        //fprintf(logfile,"Totally used time: %f second",((double)(endt - startt)) / CLOCKS_PER_SEC);
        fclose(logfile);
        
        double min; double max; Point minLoc; Point maxLoc;
        Mat t;
        minMaxLoc( cvl.layer[0].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[0].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -255.0 / min);
        cvl.layer[0].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w0_1.png", t);
        //
        minMaxLoc( cvl.layer[1].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[1].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[1].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w1_1.png",t);
        //
        minMaxLoc( cvl.layer[2].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[2].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[2].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w2_1.png",t);
        //
        minMaxLoc( cvl.layer[3].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[3].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);   
        cvl.layer[3].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w3_1.png",t);
        //
        minMaxLoc( cvl.layer[4].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[4].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[4].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w4_1.png",t);
        //
        minMaxLoc( cvl.layer[5].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[5].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[5].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w5_1.png",t);
        //
        minMaxLoc( cvl.layer[6].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[6].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[6].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w6_1.png",t);
        //
        minMaxLoc( cvl.layer[7].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[7].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[7].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w7_1.png",t);

#ifdef test_on_train_data
        /*Checking test data accuracy*/

        Mat train_result = resultPredict(x, y, cvl, HiddenLayers, smr, 3e-3, Probability);
        for (int i = 0; i < train_result.cols; i++)
        {
            printf("class :: %f\tcorrect class::%f\n", train_result.at<double>(0, i),y.at<double>(0, i));
        }
        
        Mat err1;
        y.copyTo(err1);
        err1 -= train_result;
        /*int */correct = err1.cols;
        for(int i=0; i<err1.cols; i++){
            if(err1.ATD(0, i) != 0)
            {
                --correct;
            }
        }
            
        cout << "train data cost value ::" << smr.cost << endl ;    
        cout<<"train data :: correct: "<<correct<<", total: "<<err1.cols<<", accuracy: "<<double(correct) / (double)(err1.cols)<<endl;
#endif
        ++converge;
        end_cost = clock();
        cout<<"Totally used time for this epoc: "<<((double)(end_cost - start_cost)) / CLOCKS_PER_SEC<<" second"<<endl;
        }
#endif
    }

}

Mat 
resultPredict(vector<Mat> &x, Mat &y, Cvl &cvl, vector<Ntw> &hLayers, SMR &smr, double lambda, Mat neuron_prob[]){

    int nsamples = x.size();
    vector<vector<Mat> > Conv1st;
    vector<vector<Mat> > Pool1st;
    vector<Point> PLperKernel;
    char image_name1[500];
    char image_name2[500];
    char image_name3[500];
    double min; double max; Point minLoc; Point maxLoc;
    Mat temp,tmp1,tempp;
    for(int k=0; k<nsamples; k++){
        vector<Mat> tpConv1st;
        vector<Mat> tpPool1st;
        for(int i=0; i<cvl.kernelAmount; i++){
            Mat temp = rot90(cvl.layer[i].W, 2);
            Mat tmpconv = conv2(x[k], temp, CONV_VALID);
            tmpconv += cvl.layer[i].b;

        #ifndef WRITE_TEST_IMAGE
            //tmpconv = sigmoid(tmpconv);
            tmpconv = ReLU(tmpconv);
            tpConv1st.push_back(tmpconv);
            tmpconv = Pooling(tmpconv, PoolingDim, PoolingDim, Pooling_Method, PLperKernel, true);
        #else
            sprintf(image_name1,"%d_%d_conv.png",k,i);
            minMaxLoc( tmpconv, &min, &max, &minLoc, &maxLoc, Mat() );
            tmpconv.convertTo(temp,CV_8UC1,255.0/(max-min),-255.0/min);
            imwrite(image_name1,temp);
            //sprintf(image_name1,"%d_%d_conv.png",k,i);
            //imwrite(image_name1,tmpconv);
            //tmpconv = sigmoid(tmpconv);
            tmpconv = ReLU(tmpconv);
            sprintf(image_name2,"%d_%d_Relu.png",k,i);
            minMaxLoc( tmpconv, &min, &max, &minLoc, &maxLoc, Mat() );
            tmpconv.convertTo(tempp,CV_8U,255.0/(max-min),-255.0/min);
            imwrite(image_name2,tempp);
            tpConv1st.push_back(tmpconv);
            tmpconv = Pooling(tmpconv, PoolingDim, PoolingDim, Pooling_Method, PLperKernel, true);
            sprintf(image_name3,"%d_%d_pool.png",k,i);
            minMaxLoc( tmpconv, &min, &max, &minLoc, &maxLoc, Mat() );
            tmpconv.convertTo(tmp1,CV_8U,255.0/(max-min),-255.0/min);
            imwrite(image_name3,tmp1);
        #endif
            tpPool1st.push_back(tmpconv);
        }

        Conv1st.push_back(tpConv1st);
        Pool1st.push_back(tpPool1st);
    }
    Mat convolvedX = concatenateMat(Pool1st);
    vector<Mat> acti;
    acti.push_back(convolvedX);
    Mat temp_neuron_prob[2];
    for(int i=1; i<=NumHiddenLayers; i++){
        Mat tmpacti = (hLayers[i - 1].W )*acti[i - 1] + repeat((hLayers[i - 1].b), 1, convolvedX.cols);
        #ifdef Dropout_layer
            temp_neuron_prob[i-1] = repeat(neuron_prob[i-1], 1, tmpacti.cols);
            tmpacti=tmpacti.mul(temp_neuron_prob[i-1]);
        #endif
        acti.push_back(sigmoid(tmpacti));
    }

    Mat M = smr.Weight * acti[acti.size() - 1] + repeat(smr.b, 1, nsamples);
    Mat tmp;
    reduce(M, tmp, 0, CV_REDUCE_MAX);
    M -= repeat(tmp, M.rows, 1);
    Mat p;
    exp(M, p);
    reduce(p, tmp, 0, CV_REDUCE_SUM);
    divide(p, repeat(tmp, p.rows, 1), p);
    log(p, tmp);

    Mat result = Mat::ones(1, tmp.cols, CV_64FC1);
    for(int i=0; i<tmp.cols; i++){
        double maxele = tmp.ATD(0, i);
        int which = 0;
        for(int j=1; j<tmp.rows; j++){
            if(tmp.ATD(j, i) > maxele){
                maxele = tmp.ATD(j, i);
                which = j;
            }
        }
        result.ATD(0, i) = which;
    }

    // softmax regression
    Mat groundTruth = Mat::zeros(nclasses, nsamples, CV_64FC1);
    for(int i=0; i<nsamples; i++){
        groundTruth.ATD(y.ATD(0, i), i) = 1.0;
    }

    Mat logP;
    Mat tmp2;
    log(p, logP);
    logP = logP.mul(groundTruth);
    smr.cost = - sum(logP)[0] / nsamples;
    pow(smr.Weight, 2.0, tmp2);
    smr.cost += sum(tmp2)[0] * lambda / 2;
    for(int i=0; i<cvl.kernelAmount; i++){
        pow(cvl.layer[i].W, 2.0, tmp2);
        smr.cost += sum(tmp2)[0] * lambda / 2; 
    } 

    // deconstruct
    for(int i=0; i<Conv1st.size(); i++){
        Conv1st[i].clear();
        Pool1st[i].clear();
    }
    Conv1st.clear();
    Pool1st.clear();
    acti.clear();

    return result;
}

void
saveWeight(Mat &M, string s){
    s += ".txt";
    FILE *pOut = fopen(s.c_str(), "w+");
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            double f=M.ATD(i, j);
            fprintf(pOut, "%0.18lf", f/*M.ATD(i, j)*/);
            if(j == M.cols - 1) {fprintf(pOut, "\n");}
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
  }


void readWeight(Mat M, string s){
    s += ".txt";

    std::fstream my(s.c_str(),ios::in);
    //FILE *pIn = fopen(s.c_str(), "r");
    for (int i = 0; i<M.rows; i++){
        for (int j = 0; j<M.cols; j++)
        {
            double f=0.0;

            if(my.is_open())
            {
                my>>f;
            }
            //fscanf_s(pIn, "%0.8f", &f);
            M.ATD(i, j)=f;  
            //cout << f <<endl ;
            //fgetc(pIn);
            }
        //fgetc(pIn);
    }
}

void read_batch(string filename, vector<Mat> &vec, Mat &label,char separator = ';'){
    ifstream file (filename.c_str(), ifstream::in);
    string line, path, classlabel;
    int number_of_images =0;
    if (file.is_open())
    {
        while (getline(file, line)) {
            stringstream liness(line);
            getline(liness, path, separator);
            getline(liness, classlabel);
            if(!path.empty() && !classlabel.empty()) {
                vec.push_back(imread(path, 0));
                label.ATD(0, number_of_images) = /*0*/(double)atoi(classlabel.c_str());
            }
            number_of_images++;
        }
    }
 }

void read_DB(vector<Mat> &trainX,vector<Mat> &testX, Mat &trainY, Mat &testY){

    string filename;
#ifndef SKIP_TRAIN
    filename = "train_shuffled_images_cattle.txt";
    Mat label1 = Mat::zeros(1, NUMBER_TRAIN_IMAGES, CV_64FC1);
    //Mat label1 = Mat::zeros(1, 6000, CV_64FC1);
    read_batch(filename, trainX, label1);
    label1.copyTo(trainY);
#endif
   
    filename = "test_shuffled_images_cattle.txt";
   // filename = "train_data.txt";
    Mat labelt = Mat::zeros(1, NUMBER_TEST_IMAGES, CV_64FC1);
    read_batch(filename, testX, labelt);
    labelt.copyTo(testY);
    
#ifndef SKIP_TRAIN
    for(int i = 0; i < trainX.size(); i++){
        //cvtColor(trainX[i], trainX[i], CV_RGB2YCrCb);
        trainX[i].convertTo(trainX[i], CV_64FC1, 1.0/255, 0);
    }
#endif

    for(int i = 0; i < testX.size(); i++){
        //cvtColor(testX[i], testX[i], CV_RGB2YCrCb);
        testX[i].convertTo(testX[i], CV_64FC1, 1.0/255, 0);
    }
  
#ifndef SKIP_TRAIN
    Mat tmp = concatenateMat(trainX);
#endif
    Mat tmp2 = concatenateMat(testX);

    Scalar mean;
    Scalar stddev;

#ifndef SKIP_TRAIN
    Mat alldata = Mat::zeros(tmp.rows, tmp.cols + tmp2.cols, CV_64FC1);
    tmp.copyTo(alldata(Rect(0, 0, tmp.cols, tmp.rows)));
    tmp2.copyTo(alldata(Rect(tmp.cols, 0, tmp2.cols, tmp.rows)));
    meanStdDev(alldata, mean, stddev);
#else      
    Mat alldata = Mat::zeros(tmp2.rows, tmp2.cols, CV_64FC1);
    tmp2.copyTo(alldata);
    meanStdDev(alldata, mean, stddev);
#endif

#ifndef SKIP_TRAIN
    for(int i = 0; i < trainX.size(); i++){
        divide(trainX[i] - mean, stddev, trainX[i]);
    }
#endif
   
    for(int i = 0; i < testX.size(); i++){
        divide(testX[i] - mean, stddev, testX[i]);
    }

}

int main(int argc, char** argv)
{
    long start, end, startt, endt;
    start = clock();
    srand(time(NULL));
    vector<Mat> trainX;
    vector<Mat> testX;
    Mat trainY, testY;
    
  NUMBER_TRAIN_IMAGES = atoi(argv[1]);
  NUMBER_TEST_IMAGES = atoi(argv[2]);
    
    logfile = fopen("logfile.txt","a+");

    read_DB(trainX, testX, trainY, testY);
#ifndef SKIP_TRAIN
    cout<<"Read trainX successfully, including "<<trainX[0].cols * trainX[0].rows<<" features and "<<trainX.size()<<" samples."<<endl;
    cout<<"Read trainY successfully, including "<<trainY.cols<<" samples"<<endl;
#endif
    cout<<"Read testX successfully, including "<<testX[0].cols * testX[0].rows<<" features and "<<testX.size()<<" samples."<<endl;
    cout<<"Read testY successfully, including "<<testY.cols<<" samples"<<endl;
    
    fprintf(logfile,"Read testX successfully, including %d samples\n",testX[0].cols * testX[0].rows);
    fprintf(logfile,"Read testY successfully, including %d samples\n",testY.cols);
    fclose(logfile);

#ifndef SKIP_TRAIN
    int nfeatures = trainX[0].rows * trainX[0].cols;
    int imgDim = trainX[0].rows;
    int nsamples = trainX.size();
#else
    int nfeatures = testX[0].rows * testX[0].cols;
    int imgDim = testX[0].rows;
    int nsamples = testX.size();
#endif
    Cvl cvl;
    vector<Ntw> HiddenLayers;
    SMR smr;

    ConvNetInitPrarms(cvl, HiddenLayers, smr, imgDim, nsamples);
    #ifndef SKIP_TRAIN
        // Train network using Back Propogation
        #ifndef NO_BATCH
            batch = 50; //nsamples/10;
        #else
            batch = nsamples;
        #endif

        Mat tpX = concatenateMat(trainX);
        double lrate =0.01; //getLearningRate(tpX);
        cout<<"lrate = "<<lrate<<endl;
        trainNetwork(trainX, trainY, testX, testY, cvl, HiddenLayers, smr, 3e-3, MAX_ITERATION, lrate);
        end = clock();
    #endif
    
    if(! G_CHECKING){
        // Save the trained kernels, you can load them into Matlab/GNU Octave to see what are they look like.
    #if defined(SKIP_TRAIN) || !defined(INTERMEDIATE_TEST)
        waitKey(10);
        startt = clock();
        // Test use test set
        Mat Probability[2];
        
        for (int i = 0; i < NumHiddenLayers; i++)
        {
            Probability[i] = Mat:: zeros(NumHiddenNeurons[i],1,CV_64FC1);
            char file[100];
            sprintf(file,"neuron_probability_%d",i);
            readWeight(Probability[i], file);
        }
        
        Mat result = resultPredict(testX, testY, cvl, HiddenLayers, smr, 3e-3, Probability);
        for (int i = 0; i < result.cols; i++)
        {
            double min; double max; Point minLoc; Point maxLoc;
            Mat tmp;
            printf("class :: %f\tcorrect class::%f\n", result.at<double>(0, i),testY.at<double>(0, i));
            fprintf(logfile,"class :: %f\tcorrect class::%f\n", result.at<double>(0, i),testY.at<double>(0, i));

            #ifdef SHOW_TEST_IMAGE
                minMaxLoc( testX[i], &min, &max, &minLoc, &maxLoc, Mat() );
                testX[i].convertTo(tmp,CV_8UC1,255.0 / (max - min), -(255.0 * min/ (max - min)));
                //imshow("testimage input",tmp);
                waitKey(0);
            #endif
        }
        
        //Mat err(testY);
        Mat err;
        testY.copyTo(err);
        err -= result;
        int correct = err.cols;
        for(int i=0; i<err.cols; i++){
            if(err.ATD(0, i) != 0)
            {
                --correct;
            }
        }
        cout << "test data cost value ::" << smr.cost << endl ;
        fprintf(logfile,"test data cost value :: %f\n",smr.cost);
        cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
        fprintf(logfile,"correct: %d, accuracy: %f\n", err.cols,(double(correct) / (double)(err.cols)));
        
        endt = clock();
        cout<<"Totally used time: "<<((double)(endt - startt)) / CLOCKS_PER_SEC<<" second"<<endl;
        fprintf(logfile,"Totally used time: %f second",((double)(endt - startt)) / CLOCKS_PER_SEC);
    #endif
        fclose(logfile);
    #ifndef SKIP_TRAIN
        double min; double max; Point minLoc; Point maxLoc;
        Mat t;
        minMaxLoc( cvl.layer[0].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[0].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -255.0 / min);
        cvl.layer[0].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w0_1.png", t);
        //
        minMaxLoc( cvl.layer[1].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[1].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[1].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w1_1.png",t);
        //
        minMaxLoc( cvl.layer[2].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[2].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[2].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w2_1.png",t);
        //
        minMaxLoc( cvl.layer[3].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[3].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);   
        cvl.layer[3].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w3_1.png",t);
        //
        minMaxLoc( cvl.layer[4].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[4].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[4].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w4_1.png",t);
        //
        minMaxLoc( cvl.layer[5].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[5].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[5].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w5_1.png",t);
        //
        minMaxLoc( cvl.layer[6].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[6].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[6].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w6_1.png",t);
        //
        minMaxLoc( cvl.layer[7].W, &min, &max, &minLoc, &maxLoc, Mat() );
        //cvl.layer[7].W.convertTo(t,CV_8UC1,255.0/(max-min),-255.0/min);
        cvl.layer[7].W.convertTo(t, CV_8UC1, 255.0 / (max - min), -(255.0 * min/ (max - min)));
        imwrite("w7_1.png",t);
    
         #endif
   }
    getchar();
  return 0;
}