#ifndef NN_TRAINING_DATA
#define NN_TRAINING_DATA

#include "FeedForwardNeuralNetwork.hpp"

// holds the required information for cost function and gradient calculation
struct NNTrainingData {
    int ndata; // number of data points
    int ntraining; // ntraining data points will be used for training, the rest for validation
    int xndim; // input dimension
    int yndim; // output dimension
    double ** x; // input data
    double ** y; // output data
    double *** yd1; // derivative output data
    double *** yd2; // second derivative data
    double ** w; // sqrt of data weights, i.e. 1/e_i , where e_i is the error on ith data !!! NOT 1/e_i^2 !!!
};

#endif
