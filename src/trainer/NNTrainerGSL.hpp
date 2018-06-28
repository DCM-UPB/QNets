#ifndef NN_TRAINER_GSL
#define NN_TRAINER_GSL

#include "NNTrainer.hpp"
#include "NNTrainingData.hpp"
#include "NNTrainingConfig.hpp"
#include "GSLFitStruct.hpp"
#include "FeedForwardNeuralNetwork.hpp"

#include <gsl/gsl_multifit_nlinear.h>

class NNTrainerGSL: public NNTrainer
{
protected:
    GSLFitStruct _tstruct;
    const gsl_multifit_nlinear_parameters _gsl_params;
public:
    NNTrainerGSL(const NNTrainingData &tdata, const NNTrainingConfig &tconfig, const gsl_multifit_nlinear_parameters &gsl_params = gsl_multifit_nlinear_default_parameters()): NNTrainer(tdata, tconfig), _gsl_params(gsl_params) {_tstruct.copyDatConf(_tdata, _tconfig);}
    ~NNTrainerGSL(){}

    void findFit(FeedForwardNeuralNetwork * const ffnn, double * const fit, double * const err, const int &verbose = 0);
};


#endif
