#ifndef NN_TRAINER
#define NN_TRAINER

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"
#include "NNTrainingData.hpp"
#include "NNTrainerConfig.hpp"
#include <sstream>
#include <cstddef>

struct NNTrainerStruct: public NNTrainingData, public NNTrainerConfig
{
    FeedForwardNeuralNetwork * ffnn; // Storing a pointer to the to-be-trained FFNN

    void copy(NNTrainingData * tdata)
    {
        ndata = tdata->ndata;
        ntraining = tdata->ntraining;
        xndim = tdata->xndim;
        yndim = tdata->yndim;
        x = tdata->x;
        y = tdata->y;
        yd1 = tdata->yd1;
        yd2 = tdata->yd2;
        w = tdata->w;
    }

    void copy(NNTrainerConfig * tconfig)
    {
        flag_r = tconfig->flag_r;
        flag_d1 = tconfig->flag_d1;
        flag_d2 = tconfig->flag_d2;
        lambda_r = tconfig->lambda_r;
        lambda_d1 = tconfig->lambda_d1;
        lambda_d2 = tconfig->lambda_d2;
    }
};

class NNTrainer
{
protected:
    NNTrainerStruct _tstruct;

public:
    NNTrainer(NNTrainingData * const tdata, NNTrainerConfig * const tconfig, FeedForwardNeuralNetwork * const ffnn = NULL){_tstruct.copy(tdata); _tstruct.copy(tconfig); _tstruct.ffnn = ffnn;}
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
