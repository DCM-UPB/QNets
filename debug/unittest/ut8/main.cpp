#include <iostream>
#include <cmath>
#include <assert.h>

#include "NNTrainerGSL.hpp"


double gaussian(const double x, const double a = 1., const double b = 0.) {
    return exp(-a*pow(x-b, 2));
};

// first derivative of gaussian
double gaussian_ddx(const double x, const double a = 1., const double b = 0.) {
    return 2.0*a*(b-x) * exp(-a*pow(x-b, 2));
};

// first derivative of gaussian
double gaussian_d2dx(const double x, const double a = 1., const double b = 0.) {
    return (pow(2.0*a*(b-x), 2) - 2.0*a) * exp(-a*pow(x-b, 2));
};


int main (void) {
    using namespace std;

    const bool verbose = false;
    const double TINY=0.000001;

    const int ndim = 2;
    const int xndim = ndim;
    const int nhid = 3;
    const int yndim = ndim;

    // set variational parameters in our model (which should be equal to betas after fitting)
    const double nbeta = (nhid-1) * (xndim+1) + yndim * nhid;
    const double betas[12] = {0.5, 0.5, -1., 0.2, -0.6, 0.4, 0., 0.5, -0.5, -1., 0.6, 0.4};
    assert(nbeta == 12);

    // create FFNN
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(xndim+1, nhid, yndim+1);
    ffnn->connectFFNN();
    ffnn->addVariationalFirstDerivativeSubstrate();
    ffnn->addFirstDerivativeSubstrate();
    ffnn->addSecondDerivativeSubstrate();

    ffnn->getNNLayer(0)->getNNUnit(0)->setActivationFunction(std_actf::provideActivationFunction("GSS"));
    ffnn->getNNLayer(0)->getNNUnit(1)->setActivationFunction(std_actf::provideActivationFunction("GSS"));

    ffnn->getNNLayer(1)->getNNUnit(0)->setActivationFunction(std_actf::provideActivationFunction("ID"));
    ffnn->getNNLayer(1)->getNNUnit(1)->setActivationFunction(std_actf::provideActivationFunction("ID"));

    assert(ffnn->getNBeta() == nbeta);

    // allocate data arrays
    const int ntraining = 15;
    const int nvalidation = 15;
    const int ntesting = 15;
    const int ndata = ntraining + nvalidation + ntesting;
    double ** xdata = new double*[ndata];
    double ** ydata = new double*[ndata];
    double *** d1data = new double**[ndata];
    double *** d2data = new double**[ndata];
    double ** weights = new double*[ndata];
    for (int i = 0; i<ndata; ++i) {
        xdata[i] = new double[xndim];
        ydata[i] = new double[yndim];
        weights[i] = new double[yndim];
        d1data[i] = new double*[yndim];
        d2data[i] = new double*[yndim];
        for (int j = 0; j<yndim; ++j) {
            d1data[i][j] = new double[xndim];
            d2data[i][j] = new double[xndim];
        }
    }

    // generate the data to be fitted
    const double lb = -2;
    const double ub = 2;
    random_device rdev;
    mt19937_64 rgen = std::mt19937_64(rdev());
    uniform_real_distribution<double> rd(lb,ub);
    for (int i=0; i<ndata; ++i) {
        for (int j=0; j<xndim; ++j) {
            xdata[i][j] = rd(rgen);
        }
        const double feed1 = betas[0] + betas[1] * xdata[i][0] + betas[2] * xdata[i][1];
        const double feed2 = betas[3] + betas[4] * xdata[i][0] + betas[5] * xdata[i][1];
        const double hn1 = gaussian(feed1);
        const double hn2 = gaussian(feed2);
        ydata[i][0] = betas[6] + betas[7] * hn1 + betas[8] * hn2;
        ydata[i][1] = betas[9] + betas[10] * hn1 + betas[11] * hn2;

        const double d1feed1 = gaussian_ddx(feed1);
        const double d1feed2 = gaussian_ddx(feed2);
        const double d2feed1 = gaussian_d2dx(feed1);
        const double d2feed2 = gaussian_d2dx(feed2);
        d1data[i][0][0] = (betas[7] * d1feed1 * betas[1]) + (betas[8] * d1feed2 * betas[4]);
        d1data[i][0][1] = (betas[7] * d1feed1 * betas[2]) + (betas[8] * d1feed2 * betas[5]);
        d1data[i][1][0] = (betas[10] * d1feed1 * betas[1]) + (betas[11] * d1feed2 * betas[4]);
        d1data[i][1][1] = (betas[10] * d1feed1 * betas[2]) + (betas[11] * d1feed2 * betas[5]);

        d2data[i][0][0] = (betas[7] * d2feed1 * pow(betas[1], 2)) + (betas[8] * d2feed2 * pow(betas[4], 2));
        d2data[i][0][1] = (betas[7] * d2feed1 * pow(betas[2], 2)) + (betas[8] * d2feed2 * pow(betas[5], 2));
        d2data[i][1][0] = (betas[10] * d2feed1 * pow(betas[1], 2)) + (betas[11] * d2feed2 * pow(betas[4], 2));
        d2data[i][1][1] = (betas[10] * d2feed1 * pow(betas[2], 2)) + (betas[11] * d2feed2 * pow(betas[5], 2));

        weights[i][0] = 1.0;
        weights[i][1] = 1.0;
    };

    // create data/config structs
    const int maxn_steps = 50;
    const int maxn_novali = 5;
    const int maxn_fits = 100;
    const double lambda_r = 0.000000001, lambda_d1 = 0.01, lambda_d2 = 0.01;
    NNTrainingData tdata = {ndata, ntraining, nvalidation, xndim, yndim, xdata, ydata, d1data, d2data, weights};
    NNTrainingConfig tconfig = {true, false, false, lambda_r, lambda_d1, lambda_d2, maxn_steps, maxn_novali};
    NNTrainerGSL * trainer;

    // find fit without derivs
    trainer = new NNTrainerGSL(tdata, tconfig); // NOTE: we do not normalize, to keep known beta targets
    trainer->bestFit(ffnn, maxn_fits, TINY, verbose ? 2 : 0); // fit until residual<TINY or maxn_fits reached
    assert(trainer->computeResidual(ffnn, false, true) <= TINY);
    delete trainer;

    // find fit with derivs
    ffnn->addCrossFirstDerivativeSubstrate();
    ffnn->addCrossSecondDerivativeSubstrate();
    tconfig.flag_d1 = true;
    tconfig.flag_d2 = true;
    trainer = new NNTrainerGSL(tdata, tconfig);
    trainer->bestFit(ffnn, maxn_fits, TINY, verbose ? 2 : 0);
    assert(trainer->computeResidual(ffnn, false, true) <= TINY);
    delete trainer;

    //ffnn->storeOnFile("nn.txt");

    // Delete allocations
    delete ffnn;

    for (int i = 0; i<ndata; ++i) {
        delete [] xdata[i];
        delete [] ydata[i];
        delete [] weights[i];
        for (int j = 0; j<yndim; ++j) {
            delete [] d1data[i][j];
            delete [] d2data[i][j];
        }
        delete [] d1data[i];
        delete [] d2data[i];
    }
    delete [] xdata;
    delete [] ydata;
    delete [] weights;
    delete [] d1data;
    delete [] d2data;

    return 0;
    //
}
