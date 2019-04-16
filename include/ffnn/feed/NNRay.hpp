#ifndef FFNN_FEED_NNRAY_HPP
#define FFNN_FEED_NNRAY_HPP

#include "ffnn/feed/WeightedFeeder.hpp"
#include "ffnn/layer/NetworkLayer.hpp"

#include <string>

class NNRay: public WeightedFeeder
{
public:
    explicit NNRay(NetworkLayer * nl);
    ~NNRay() override = default;

    // string code methods
    std::string getIdCode() override { return "RAY"; } // return an identification string
    void setParams(const std::string &params) override;

    // variational parameters
    int setVariationalParametersIndexes(const int &starting_index, bool flag_add_vp = true) override;

    // return the feed mean value (mu) and standard deviation (sigma)
    double getFeedMu() override;
    double getFeedSigma() override;

    // Computation
    double getFeed() override;
    double getFirstDerivativeFeed(const int &i1d) override;
    double getSecondDerivativeFeed(const int &i2d) override;
    double getVariationalFirstDerivativeFeed(const int &iv1d) override;
    double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d) override;
    double getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d) override;

    // randomizer implementations
    void randomizeBeta() override;
};

#endif
