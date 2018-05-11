#ifndef NN_TRAINER_GSL
#define NN_TRAINER_GSL

#include "NNTrainer.hpp"
#include "NNTrainingDataGSL.hpp"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>


class NNTrainerGSL: public NNTrainer
{
protected:
    NNTrainingDataGSL * _tdata;

public:
    //NNTrainerGSL(NNTrainingDataGSL * tdata, FeedForwardNeuralNetwork * ffnn) {_tdata = tdata; _ffnn = ffnn;};

    void findFit();

};


#endif
