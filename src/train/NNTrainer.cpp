#include "ffnn/train/NNTrainer.hpp"
#include "ffnn/feed/SmartBetaGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <random>

// --- Helpers

inline double computeMu(const double * const * const array, const int &len, const int &index)
{
    double mean = 0.;
    for (int i=0; i<len; ++i) { mean += array[i][index];
}
    return mean/len;
}

inline double computeSigma(const double * const * const array, const int &len, const int &index, const double &mean)
{
    double std = 0.;
    for (int i=0; i<len; ++i) { std += pow(array[i][index] - mean , 2);
}
    return sqrt(std/(len-1));
}

inline double computeSigma(const double * const * const array, const int &len, const int &index)
{
    return computeSigma(array, len, index, computeMu(array, len, index));
}

inline void computeBounds(const double * const * const array, const int &len, const int &index, double &lbound, double &ubound)
{
    lbound = array[0][index];
    ubound = array[0][index];
    for (int i=1; i<len; ++i) {
        if (array[i][index] < lbound) { lbound = array[i][index];
        } else if (array[i][index] > ubound) { ubound = array[i][index];
}
    }
}


// -- Class methods

FeedForwardNeuralNetwork * NNTrainer::_createVDerivFFNN(FeedForwardNeuralNetwork * const ffnn)
{
    auto * ffnn_vderiv = new FeedForwardNeuralNetwork(ffnn);
    _configureFFNN(ffnn_vderiv, true);

    return ffnn_vderiv;
}

void NNTrainer::_configureFFNN(FeedForwardNeuralNetwork * const ffnn, const bool flag_vderiv)
{
    if (!ffnn->isConnected()) { // if the user didn't connect, we default for him
        ffnn->connectFFNN();
        ffnn->assignVariationalParameters();
    }
    const bool flag_cd1 = _flag_d1 || _flag_d2; // second cross derivative also needs first one
    if (flag_vderiv) { ffnn->addSubstrates(flag_cd1, _flag_d2, true, flag_cd1, _flag_d2);
    } else { ffnn->addSubstrates(flag_cd1, _flag_d2, false, false, false);
}
}


void NNTrainer::setNormalization(FeedForwardNeuralNetwork * const ffnn)
{
    // input side
    for (int i=0; i<_tdata.xndim; ++i) {
        double mu = computeMu(_tdata.x, _tdata.ndata, i);
        ffnn->getInputLayer()->getInputUnit(i)->setInputMu(mu);
        ffnn->getInputLayer()->getInputUnit(i)->setInputSigma(computeSigma(_tdata.x, _tdata.ndata, i, mu));
    }

    // output side
    for (int i=0; i<_tdata.yndim; ++i) {
        double lbound, ubound;
        computeBounds(_tdata.y, _tdata.ndata, i, lbound, ubound);
        ffnn->getOutputLayer()->getOutputNNUnit(i)->setOutputBounds(lbound, ubound);
    }
}

double NNTrainer::computeResidual(FeedForwardNeuralNetwork * const ffnn, const bool &flag_r, const bool &flag_d)
{
    //
    // Basic form is sqrt(1/N sum (f(x) - y)^2), but inside the sqrt additional terms may be added
    //
    const int npar = ffnn->getNVariationalParameters();
    const int offset = _flag_test ? _tdata.ntraining + _tdata.nvalidation : 0; // if no testing data, we fall back to the full training + vali set
    const double scale = _flag_test ? 1./(_tdata.ndata - offset) : 1./(_tdata.ntraining + _tdata.nvalidation);
    const double lambda_r_fac = _tconfig.lambda_r * _tconfig.lambda_r / npar;
    const double lambda_d1_fac = scale*_tconfig.lambda_d1 * _tconfig.lambda_d1 / _tdata.xndim;
    const double lambda_d2_fac = scale*_tconfig.lambda_d2 * _tconfig.lambda_d2 / _tdata.xndim;


    double resi = 0.;

    if (_flag_r && flag_r) {
        // add regularization residual from NN betas
        for (int i=0; i<npar; ++i){
            resi += lambda_r_fac * pow(ffnn->getBeta(i), 2);
        }
    }

    //get difference NN vs data
    for (int i=offset; i<_tdata.ndata; ++i) {
        ffnn->setInput(_tdata.x[i]);
        ffnn->FFPropagate();
        for (int j=0; j<_tdata.yndim; ++j) {
            resi += scale * pow(_tdata.w[i][j] * (ffnn->getOutput(j) - _tdata.y[i][j]), 2);

            if (flag_d) { // add derivative residuals
                for (int k=0; k<_tdata.xndim; ++k) {
                    if (_flag_d1) { resi += lambda_d1_fac * pow(_tdata.w[i][j] * (ffnn->getFirstDerivative(j, k) - _tdata.yd1[i][j][k]), 2);
}
                    if (_flag_d2) { resi += lambda_d2_fac * pow(_tdata.w[i][j] * (ffnn->getSecondDerivative(j, k) - _tdata.yd2[i][j][k]), 2);
}
                }
            }
        }
    }
    return sqrt(resi);
}

void NNTrainer::bestFit(FeedForwardNeuralNetwork * const ffnn, double * bestfit, double * bestfit_err, const int &nfits, const double &resi_target, const int &verbose, const bool &flag_smart_beta)
{
    int npar = ffnn->getNVariationalParameters();
    double fit[npar], err[npar];
    double bestresi_pure = -1.0, bestresi_noreg = -1.0, bestresi_full = -1.0;

    if (!_flag_test && verbose > 0) { fprintf(stderr, "[NNTrainer] Warning: Testing residual calculation disabled, i.e. testing is based on training+validation data.\n");
}

    _configureFFNN(ffnn, false); // set non-vderiv substrates (the child implementation may use a copy FFNN with variational substrates)

    int ifit = 0;
    while(true) {
        // initial parameters
        if (flag_smart_beta) { smart_beta::generateSmartBeta(ffnn);
        } else if (ffnn->getNFeatureMapLayers() > 0) { // hack because of fitting problems when using FMLs
            random_device rdev;
            mt19937_64 rgen = std::mt19937_64(rdev());
            uniform_real_distribution<double> rd(-0.1,0.1);
            for (int i=0; i<ffnn->getNBeta(); ++i) { ffnn->setBeta(i, rd(rgen));
}
        }
        else { ffnn->randomizeBetas();
}

        findFit(ffnn, fit, err, verbose); // try new fit
        ffnn->setVariationalParameter(fit); // make sure ffnn is set to fit betas

        double resi_full = computeResidual(ffnn, true, true);
        double resi_noreg = computeResidual(ffnn, false, true);
        double resi_pure = computeResidual(ffnn, false, false);

        // check for new best testing residual
        if(ifit < 1 || (resi_noreg < bestresi_noreg)) {
            for(int i = 0; i<npar; ++i){
                bestfit[i] = fit[i];
                bestfit_err[i] = err[i];
            }
            bestresi_full = resi_full;
            bestresi_noreg = resi_noreg;
            bestresi_pure = resi_pure;
        }

        ++ifit;

        // check break conditions
        if (bestresi_noreg <= resi_target) {
            if (verbose > 0) { fprintf(stderr, "Unregularized testing residual %f (full: %f, pure: %f) meets tolerance %f. Exiting with good fit.\n\n", bestresi_noreg, bestresi_full, bestresi_pure, resi_target);
}
            break;
        } 
            if (verbose > 0) { fprintf(stderr, "Unregularized testing residual %f (full: %f, pure: %f) above tolerance %f.\n", resi_noreg, resi_full, resi_pure, resi_target);
}
            if (ifit >= nfits) {
                if (verbose > 0) { fprintf(stderr, "Maximum number of fits reached (%i). Exiting with best unregularized testing residual %f.\n\n", nfits, bestresi_noreg);
}
                break;
            }
            if (verbose > 0) { fprintf(stderr, "Let's try again.\n");
}
        
    }

    if (verbose > 0) { // print summary
        fprintf(stderr, "best fit summary:\n");
        for(int i=0; i<npar; ++i) { fprintf(stderr, "b%i      = %.5f +/- %.5f\n", i, bestfit[i], bestfit_err[i]);
}
        fprintf(stderr, "|f(x)| = %f (w/o reg: %f, pure: %f)\n\n", bestresi_full, bestresi_noreg, bestresi_pure);
    }

    // set ffnn to bestfit betas
    ffnn->setVariationalParameter(bestfit);
}

void NNTrainer::bestFit(FeedForwardNeuralNetwork * const ffnn, const int &nfits, const double &resi_target, const int &verbose, const bool &flag_smart_beta)
{
    double bestfit[ffnn->getNVariationalParameters()], bestfit_err[ffnn->getNVariationalParameters()];
    bestFit(ffnn, bestfit, bestfit_err, nfits, resi_target, verbose, flag_smart_beta);
}

// print output of fitted NN to file
void NNTrainer::printFitOutput(FeedForwardNeuralNetwork * const ffnn, const double &min, const double &max, const int &npoints, const bool &print_d1, const bool &print_d2, const double * base_input)
{
    using namespace std;
    const double default_base_input = 0.;
    const double * my_base_input;
    if (base_input != nullptr) { my_base_input = base_input;
    } else { my_base_input = &default_base_input;
}

    // add required substrates if necessary
    if ((print_d1 || print_d2) && !ffnn->hasFirstDerivativeSubstrate()) { ffnn->addFirstDerivativeSubstrate();
}
    if (print_d2 && !ffnn->hasSecondDerivativeSubstrate()) { ffnn->addSecondDerivativeSubstrate();
}

    for (int i = 0; i<_tdata.xndim; ++i) {
        for (int j = 0; j<_tdata.yndim; ++j) {
            stringstream ss;
            ss << i << "_" << j << ".txt";
            writePlotFile(ffnn, my_base_input, i, j, min, max, npoints, "getOutput", "v_" + ss.str());
            if (print_d1) { writePlotFile(ffnn, my_base_input, i, j, min, max, npoints, "getFirstDerivative", "d1_" + ss.str());
}
            if (print_d2) { writePlotFile(ffnn, my_base_input, i, j, min, max, npoints, "getSecondDerivative", "d2_" + ss.str());
}
        }
    }
}
