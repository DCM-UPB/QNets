#ifndef GSL_FIT_STRUCT
#define GSL_FIT_STRUCT

#include "NNTrainingData.hpp"
#include "NNTrainingConfig.hpp"
#include "FeedForwardNeuralNetwork.hpp"

#include <gsl/gsl_vector.h>
#include <cstddef> // NULL

// full struct for training, to be passed to NNTrainer and subsequently to e.g. GSL methods
struct GSLFitStruct: public NNTrainingData, public NNTrainingConfig
{
    FeedForwardNeuralNetwork * ffnn = NULL; // Storing a pointer to the to-be-trained FFNN

    // validation residuals
    gsl_vector * fvali_full = NULL;
    gsl_vector * fvali_noreg = NULL;
    gsl_vector * fvali_pure = NULL;

    void copyData(const NNTrainingData &tdata)
    {
        ndata = tdata.ndata;
        ntraining = tdata.ntraining;
        nvalidation = tdata.nvalidation;
        xndim = tdata.xndim;
        yndim = tdata.yndim;
        x = tdata.x;
        y = tdata.y;
        yd1 = tdata.yd1;
        yd2 = tdata.yd2;
        w = tdata.w;
    }

    void copyConfig(const NNTrainingConfig &tconfig)
    {
        flag_r = tconfig.flag_r;
        flag_d1 = tconfig.flag_d1;
        flag_d2 = tconfig.flag_d2;
        lambda_r = tconfig.lambda_r;
        lambda_d1 = tconfig.lambda_d1;
        lambda_d2 = tconfig.lambda_d2;
        maxn_steps = tconfig.maxn_steps;
        maxn_novali = tconfig.maxn_novali;
    }

    void copyDatConf(const NNTrainingData &tdata, const NNTrainingConfig &tconfig)
    {
        copyData(tdata);
        copyConfig(tconfig);
    }
};

#endif
