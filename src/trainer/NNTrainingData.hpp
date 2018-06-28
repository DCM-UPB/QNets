#ifndef NN_TRAINING_DATA
#define NN_TRAINING_DATA

// holds the required information for cost function and gradient calculation
struct NNTrainingData {
    int ndata; // number of data points
    int ntraining; // ntraining data points will be used as training set, the rest for testing set
    int nvalidation; // out of the ntraining training data, nvalidation points will be held back as validation set
    int xndim; // input dimension
    int yndim; // output dimension
    double ** x; // input data, shape (ndata, xndim)
    double ** y; // output data, shape (ndata, yndim)
    double *** yd1; // derivative output data, shape (ndata, yndim, xndim)
    double *** yd2; // second derivative data, shape (ndata, yndim, xndim)
    double ** w; // sqrt of y data weights, i.e. 1/e_i , where e_i is the error on ith data !!! NOT 1/e_i^2 !!!
};

#endif
