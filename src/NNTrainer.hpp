#ifndef NN_TRAINER
#define NN_TRAINER

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"
#include "NNTrainingData.hpp"
#include "NNTrainerConfig.hpp"
#include "NNTrainerStruct.hpp"
#include <sstream>
#include <cstddef>


class NNTrainer
{
protected:
    NNTrainerStruct _tstruct;
public:
    // construct from individual structures / ffnn
    NNTrainer(NNTrainingData * const tdata, NNTrainerConfig * const tconfig, FeedForwardNeuralNetwork * const ffnn = NULL)
    {
        _tstruct.copyData(tdata); _tstruct.copyConfig(tconfig); _tstruct.ffnn = ffnn;
    }

    // construct from full trainer struct
    NNTrainer(NNTrainerStruct * const tstruct){_tstruct = *tstruct;}

    virtual ~NNTrainer(){}

    // find best fit from a number of nfits fits
    void bestFit(const int nsteps, const int nfits, const double tolresi, const int verbose);

    // print output of fitted NN to file
    void printFitOutput(const double min, const double max, const int npoints, const double xscale, const double yscale, const double xshift, const double yshift, const bool print_d1 = false, const bool print_d2 = false);

    // store fitted NN in file
    void printFitNN() {_tstruct.ffnn->storeOnFile("nn.txt");}

    virtual void findFit(double * const fit, double * const err, double &resi_full, double &resi_noreg, double &resi_pure, const int nsteps, const int verbose) = 0; // to be implemented by child
};


#endif
