#ifndef FFNN_TRAIN_NNTRAINER_HPP
#define FFNN_TRAIN_NNTRAINER_HPP

#include "qnets/poly/io/PrintUtilities.hpp"
#include "qnets/poly/FeedForwardNeuralNetwork.hpp"
#include "qnets/poly/train/NNTrainingConfig.hpp"
#include "qnets/poly/train/NNTrainingData.hpp"
#include <cstddef> // NULL
#include <sstream>


class NNTrainer
{
protected:
    const NNTrainingData _tdata; // holds all the input data, their counts and weights
    const NNTrainingConfig _tconfig; // holds training parameters and flags, mainly to configure the residual
    const bool _flag_vali; // do we have validation data?
    const bool _flag_test; // do we have testing data?
    const bool _flag_r; // lambda_r > 0 ?
    const bool _flag_d1; // lambda_d1 > 0 ?
    const bool _flag_d2; // lambda_d2 > 0 ?

    // return a copy of ffnn with enabled vderiv substrates
    FeedForwardNeuralNetwork * _createVDerivFFNN(FeedForwardNeuralNetwork * ffnn);

    // connect and add necessary substrates to ffnn (vderiv substrates only if flag is true)
    void _configureFFNN(FeedForwardNeuralNetwork * ffnn, bool flag_vderiv = false);
public:
    // construct from individual structures / ffnn
    NNTrainer(const NNTrainingData &tdata, const NNTrainingConfig &tconfig)
            :
            _tdata(tdata), _tconfig(tconfig), _flag_vali(tdata.nvalidation > 0), _flag_test((tdata.ntraining + tdata.nvalidation) < tdata.ndata),
            _flag_r(tconfig.lambda_r > 0), _flag_d1(tconfig.lambda_d1 > 0), _flag_d2(tconfig.lambda_d2 > 0) {}

    virtual ~NNTrainer() = default;

    // set shift/scale parameters of NN units, to achieve proper normalization with respect to tdata
    void setNormalization(FeedForwardNeuralNetwork * ffnn);

    // compute testing residual of ffnn vs testing data in _tdata (vs training+validation if no testing present)
    double computeResidual(FeedForwardNeuralNetwork * ffnn, const bool &flag_r = false, const bool &flag_d = false);

    // find individual fit, to be implemented by child
    virtual void findFit(FeedForwardNeuralNetwork * ffnn, double * fit, double * err, const int &verbose = 0) = 0;

    // find best fit from a number of nfits fits
    void bestFit(FeedForwardNeuralNetwork * ffnn, double * bestfit, double * bestfit_err, const int &nfits, const double &resi_target = 0., const int &verbose = 0, const bool &flag_smart_beta = false); // fits NN and provides best betas with errors in bestfit(_err)
    void bestFit(FeedForwardNeuralNetwork * ffnn, const int &nfits, const double &resi_target = 0., const int &verbose = 0, const bool &flag_smart_beta = false); // fits NN and uses internal bestfit(_err) arrays

    // print output of fitted NN to file
    void printFitOutput(FeedForwardNeuralNetwork * ffnn, const double &min, const double &max, const int &npoints, const bool &print_d1 = false, const bool &print_d2 = false, const double * base_input = nullptr);
};


#endif
