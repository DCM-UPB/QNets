#ifndef NN_TRAINING_DATA
#define NN_TRAINING_DATA

#include "FeedForwardNeuralNetwork.hpp"

// holds the required information for cost function and gradient calculation
struct NNTrainingData {
    const int n; // number of data points
    const int xdim; // input dimension
    const int ydim; // output dimension
    const double * const * const x; // input data
    const double * const * const y; // output data
    const double * const * const * const yd1; // derivative output data
    const double * const * const * const yd2; // second derivative data
    const double * const * const w; // sqrt of data weights, i.e. 1/e_i , where e_i is the error on ith data !!! NOT 1/e_i^2 !!!
    const double lambda_d1, lambda_d2, lambda_r; // derivative and regularization weights
    const bool flag_d1, flag_d2, flag_r; // use derivatives / regularization?
};

#endif
