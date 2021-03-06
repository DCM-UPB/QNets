#ifndef FFNN_TRAIN_NNTRAININGDATA_HPP
#define FFNN_TRAIN_NNTRAININGDATA_HPP

#include <cstddef> // NULL

// holds the required information for cost function and gradient calculation
struct NNTrainingData
{
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

    void allocate(const bool flag_d1 = false, const bool flag_d2 = false) // allocate data arrays
    {
        deallocate(); // prevent user from producing memleaks

        x = new double * [ndata];
        y = new double * [ndata];
        w = new double * [ndata];
        if (flag_d1) {
            yd1 = new double ** [ndata];
        }
        if (flag_d2) {
            yd2 = new double ** [ndata];
        }
        for (int i = 0; i < ndata; ++i) {
            x[i] = new double[xndim];
            y[i] = new double[yndim];
            w[i] = new double[yndim];
            if (flag_d1) {
                yd1[i] = new double * [yndim];
            }
            if (flag_d2) {
                yd2[i] = new double * [yndim];
            }
            for (int j = 0; j < yndim; ++j) {
                if (flag_d1) {
                    yd1[i][j] = new double[xndim];
                }
                if (flag_d2) {
                    yd2[i][j] = new double[xndim];
                }
            }
        }
    }

    void deallocate() // deallocate data arrays
    {
        for (int i = 0; i < ndata; ++i) {
            if (x != nullptr) { delete[] x[i]; }
            if (y != nullptr) { delete[] y[i]; }
            if (w != nullptr) { delete[] w[i]; }
            for (int j = 0; j < yndim; ++j) {
                if (yd1 != nullptr) { delete[] yd1[i][j]; }
                if (yd2 != nullptr) { delete[] yd2[i][j]; }
            }
            if (yd1 != nullptr) { delete[] yd1[i]; }
            if (yd2 != nullptr) { delete[] yd2[i]; }
        }
        delete[] x;
        delete[] y;
        delete[] w;
        delete[] yd1;
        delete[] yd2;

        x = nullptr;
        y = nullptr;
        w = nullptr;
        yd1 = nullptr;
        yd2 = nullptr;
    }
};

#endif
