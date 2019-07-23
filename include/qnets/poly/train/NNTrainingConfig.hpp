#ifndef FFNN_TRAIN_NNTRAININGCONFIG_HPP
#define FFNN_TRAIN_NNTRAININGCONFIG_HPP

#include "qnets/poly/FeedForwardNeuralNetwork.hpp"

// holds the required configuration parameters for the trainer
struct NNTrainingConfig
{
    double lambda_r, lambda_d1, lambda_d2; // regularization / derivative weights
    int maxn_steps, maxn_novali; // maximum number of fitting iterations / stop early after how many steps without improved validation (0 -> disabled)
};

#endif
