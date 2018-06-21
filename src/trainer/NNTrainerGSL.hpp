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
    NNTrainerGSL(const NNTrainingData &tdata, const NNTrainingConfig &tconfig, FeedForwardNeuralNetwork * const ffnn = NULL): NNTrainer(tdata, tconfig, ffnn)
    {
        _tstruct.copyData(_tdata); _tstruct.copyConfig(_tconfig); _tstruct.ffnn = _ffnn;
    };

    void findFit(double * const fit, double * const err, double &resi_full, double &resi_noreg, double &resi_pure, const int &nsteps, const int &verbose);
};


#endif
