#ifndef NN_TRAINER
#define NN_TRAINER

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"
#include "NNTrainingData.hpp"
#include "NNTrainingConfig.hpp"
#include <sstream>
#include <cstddef>


class NNTrainer
{
protected:
    const NNTrainingData _tdata; // holds all the input data, their counts and weights
    const NNTrainingConfig _tconfig; // holds training parameters and flags, mainly to configure the residual
    const bool _flag_vali; // do we have validation data?
    const bool _flag_test; // do we have testing data?
public:
    // construct from individual structures / ffnn
    NNTrainer(const NNTrainingData &tdata, const NNTrainingConfig &tconfig) : _tdata(tdata), _tconfig(tconfig), _flag_vali(_tdata.nvalidation > 0), _flag_test(_tdata.ntraining + _tdata.nvalidation == _tdata.ndata) {}

    virtual ~NNTrainer(){}

    // set shift/scale parameters of NN units, to achieve proper normalization with respect to tdata
    void setNormalization(FeedForwardNeuralNetwork * const ffnn);

    // compute residual of ffnn vs data in _tdata
    double computeResidual(FeedForwardNeuralNetwork * const ffnn, const bool &flag_r = false, const bool &flag_d = false);

    // find individual fit, to be implemented by child
    virtual void findFit(FeedForwardNeuralNetwork * const ffnn, double * const fit, double * const err, const int &verbose = 0) = 0;

    // find best fit from a number of nfits fits
    void bestFit(FeedForwardNeuralNetwork * const ffnn, double * const bestfit, double * const bestfit_err, const int &nfits, const double &resi_target = 0., const int &verbose = 0); // fits NN and provides best betas with errors in bestfit(_err)
    void bestFit(FeedForwardNeuralNetwork * const ffnn, const int &nfits, const double &resi_target = 0., const int &verbose = 0); // fits NN and uses internal bestfit(_err) arrays

    // print output of fitted NN to file
    void printFitOutput(FeedForwardNeuralNetwork * const ffnn, const double &min, const double &max, const int &npoints, const bool &print_d1 = false, const bool &print_d2 = false);
};


#endif
