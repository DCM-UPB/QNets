#ifndef SMART_BETA_GENERATOR
#define SMART_BETA_GENERATOR

#include "FeedForwardNeuralNetwork.hpp"
#include "FedLayer.hpp"
#include "FeederInterface.hpp"
#include "NNRay.hpp"


namespace smart_beta {

    namespace details {
        // --- Internal details, exposed for unit testing

        const double MIN_BETA_NORM = 0.001; // minimal allowed beta vector norm
        const int N_TRY_BEST_LD_BETA = 20; // how many tries to find best ld beta
        const int BETA_INDEX_OFFSET = 0; // leave out beta before index (experimental, don't use)

        NNRay * _castFeederToRay(FeederInterface * const feeder);
        std::vector<int> _findIndexesOfUnitsWithRay(FedLayer * L);
        void _computeBetaMuAndSigma(FedUnit * U, double &mu, double &sigma);
        void _setRandomBeta(FeederInterface * ray, const double &mu, const double &sigma);
        void _makeBetaOrthogonal(FeederInterface * fixed_ray, FeederInterface * ray);
    }

    // generate and set smart betas for the Layer L
    void generateSmartBeta(FeedForwardNeuralNetwork * ffnn);
    void generateSmartBeta(FedLayer * L);

}


#endif
