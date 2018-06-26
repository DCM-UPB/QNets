#ifndef SMART_BETA_GENERATOR
#define SMART_BETA_GENERATOR

#include "FeedForwardNeuralNetwork.hpp"
#include "FedNetworkLayer.hpp"


namespace smart_beta{

    // generate and set smart betas for the Layer L
    void generateSmartBeta(FeedForwardNeuralNetwork * ffnn);
    void generateSmartBeta(FedNetworkLayer * L);

}


#endif
