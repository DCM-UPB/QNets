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
    FeedForwardNeuralNetwork * _ffnn;
public:
    // construct from individual structures / ffnn
    NNTrainer(const NNTrainingData &tdata, const NNTrainingConfig &tconfig, FeedForwardNeuralNetwork * const ffnn = NULL)
    {
        _tdata = tdata; _tconfig = tconfig; _ffnn = ffnn;
    }

    virtual ~NNTrainer(){}

    virtual void findFit(double * const fit, double * const err, const int &maxnsteps, const int &verbose) = 0; // to be implemented by child

    // find best fit from a number of nfits fits
    void bestFit(const int &maxnsteps, const int &nfits, const double &resi_target, const int &verbose);

    // print output of fitted NN to file
    void printFitOutput(const double &min, const double &max, const int &npoints, const double &xscale, const double &yscale, const double &xshift, const double &yshift, const bool &print_d1 = false, const bool &print_d2 = false);

    // store fitted NN in file
    void printFitNN() {_ffnn->storeOnFile("nn.txt");}
};


#endif
