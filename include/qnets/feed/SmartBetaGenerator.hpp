#ifndef FFNN_FEED_SMARTBETAGENERATOR_HPP
#define FFNN_FEED_SMARTBETAGENERATOR_HPP

#include "qnets/feed/FeederInterface.hpp"
#include "qnets/feed/NNRay.hpp"
#include "qnets/layer/FedLayer.hpp"
#include "qnets/net/FeedForwardNeuralNetwork.hpp"


namespace smart_beta
{

namespace details
{
// --- Internal details, exposed for unit testing

const double MIN_BETA_NORM = 0.001; // minimal allowed beta vector norm
const int N_TRY_BEST_LD_BETA = 20; // how many tries to find best ld beta
const int BETA_INDEX_OFFSET = 0; // leave out beta before index (experimental, don't use)

NNRay * _castFeederToRay(FeederInterface * feeder);
std::vector<int> _findIndexesOfUnitsWithRay(FedLayer * L);
void _computeBetaMuAndSigma(FedUnit * U, double &mu, double &sigma);
void _setRandomBeta(FeederInterface * ray, const double &mu, const double &sigma);
void _makeBetaOrthogonal(FeederInterface * fixed_ray, FeederInterface * ray);
}  // namespace details

// generate and set smart betas for the Layer L
void generateSmartBeta(FeedForwardNeuralNetwork * ffnn);
void generateSmartBeta(FedLayer * L);
}  // namespace smart_beta


#endif
