// holds the required information for cost function and gradient calculation
struct NNTrainingData {
    const int n; // number of data
    const int xdim; // input dimension
    const int ydim; // output dimension
    double ** x; // input data
    double ** y; // output data
    double *** yd1; // derivative output data
    double *** yd2; // second derivative data
    double **w; // sqrt of data weights 1/e_i , where e_i is the error on ith data !!! NOT 1/e_i^2 !!!
    double lambda_d1, lambda_d2, lambda_r; // derivative and regularization weights
    bool flag_d1, flag_d2, flag_r; // use first/second derivatives / regularization?
    FeedForwardNeuralNetwork * ffnn;
};
