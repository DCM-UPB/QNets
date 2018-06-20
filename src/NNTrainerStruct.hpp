#ifndef NN_TRAINER_STRUCT
#define NN_TRAINER_STRUCT

#include "NNTrainingData.hpp"
#include "NNTrainerConfig.hpp"
#include "FeedForwardNeuralNetwork.hpp"

// full struct for training, to be passed to NNTrainer and subsequently to e.g. GSL methods
struct NNTrainerStruct: public NNTrainingData, public NNTrainerConfig
{
    FeedForwardNeuralNetwork * ffnn; // Storing a pointer to the to-be-trained FFNN

    void copyData(NNTrainingData * tdata)
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

    void copyConfig(NNTrainerConfig * tconfig)
    {
        flag_r = tconfig->flag_r;
        flag_d1 = tconfig->flag_d1;
        flag_d2 = tconfig->flag_d2;
        lambda_r = tconfig->lambda_r;
        lambda_d1 = tconfig->lambda_d1;
        lambda_d2 = tconfig->lambda_d2;
    }
};

#endif
