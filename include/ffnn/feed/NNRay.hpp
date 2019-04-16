#ifndef FFNN_FEED_NNRAY_HPP
#define FFNN_FEED_NNRAY_HPP

#include "ffnn/feed/WeightedFeeder.hpp"
#include "ffnn/layer/NetworkLayer.hpp"

#include <string>

class NNRay: public WeightedFeeder
{
public:
    explicit NNRay(NetworkLayer * nl);
    ~NNRay() final = default;

    // string code methods
    std::string getIdCode() final { return "RAY"; } // return an identification string
    void setParams(const std::string &params) final;

    // variational parameters
    int setVariationalParametersIndexes(const int &starting_index, bool flag_add_vp = true) final;

    // return the feed mean value (mu) and standard deviation (sigma)
    double getFeedMu() final;
    double getFeedSigma() final;

    // Computation
    double getFeed() final;
    double getFirstDerivativeFeed(const int &i1d) final;
    double getSecondDerivativeFeed(const int &i2d) final;
    double getVariationalFirstDerivativeFeed(const int &iv1d) final;
    double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d) final;
    double getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d) final;

    // randomizer implementations
    void randomizeBeta() final;
};

#endif
