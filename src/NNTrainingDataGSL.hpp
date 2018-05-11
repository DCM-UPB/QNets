#ifndef NN_TRAINING_DATA_GSL
#define NN_TRAINING_DATA_GSL

#include "NNTrainingData.hpp"

struct NNTrainingDataGSL: public NNTrainingData
{
    FeedForwardNeuralNetwork * ffnn; // Storing a pointer to the FFNN, needed for GSL calls
};

#endif
