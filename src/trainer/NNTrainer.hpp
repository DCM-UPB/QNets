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
    NNTrainingData _tdata;
    NNTrainingConfig _tconfig;
public:
    // construct from individual structures / ffnn
    NNTrainer(const NNTrainingData &tdata, const NNTrainingConfig &tconfig){_tdata = tdata; _tconfig = tconfig;}

    virtual ~NNTrainer(){}

    // compute residual of ffnn vs data in _tdata
    double computeResidual(FeedForwardNeuralNetwork * const ffnn, const bool &flag_r = false, const bool &flag_d = false);

    // find individual fit, to be implemented by child
    virtual void findFit(FeedForwardNeuralNetwork * const ffnn, double * const fit, double * const err, const int &maxnsteps, const int &verbose) = 0;

    // find best fit from a number of nfits fits
    void bestFit(FeedForwardNeuralNetwork * const ffnn, double * const bestfit, double * const bestfit_err, const int &maxnsteps, const int &nfits, const double &resi_target = 0., const int &verbose = 0); // fits NN and provides best betas with errors in bestfit(_err)
    void bestFit(FeedForwardNeuralNetwork * const ffnn, const int &maxnsteps, const int &nfits, const double &resi_target = 0., const int &verbose = 0); // fits NN and uses internal bestfit(_err) arrays

    // print output of fitted NN to file
    void printFitOutput(FeedForwardNeuralNetwork * const ffnn, const double &min, const double &max, const int &npoints, const double &xscale, const double &yscale, const double &xshift, const double &yshift, const bool &print_d1 = false, const bool &print_d2 = false);
};


#endif
