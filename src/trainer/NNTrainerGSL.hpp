#ifndef NN_TRAINER_GSL
#define NN_TRAINER_GSL

#include "NNTrainer.hpp"
#include "NNTrainingData.hpp"
#include "NNTrainingConfig.hpp"
#include "GSLFitStruct.hpp"
#include "FeedForwardNeuralNetwork.hpp"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>


class NNTrainerGSL: public NNTrainer
{
protected:
    GSLFitStruct _tstruct;
public:
    NNTrainerGSL(const NNTrainingData &tdata, const NNTrainingConfig &tconfig): NNTrainer(tdata, tconfig)
    {
        _tstruct.copyData(_tdata); _tstruct.copyConfig(_tconfig);
    };

    void findFit(FeedForwardNeuralNetwork * const ffnn, double * const fit, double * const err, const int &maxnsteps, const int &verbose);
};


#endif
