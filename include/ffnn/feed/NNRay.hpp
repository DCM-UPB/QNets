#ifndef NN_RAY
#define NN_RAY

#include "ffnn/feed/WeightedFeeder.hpp"
#include "ffnn/layer/NetworkLayer.hpp"

#include <string>

class NNRay: public WeightedFeeder
{
public:
    explicit NNRay(NetworkLayer * nl);
    ~NNRay(){}

    // string code methods
    std::string getIdCode(){return "RAY";} // return an identification string
    void setParams(const std::string &params);

    // variational parameters
    int setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp = true);

    // return the feed mean value (mu) and standard deviation (sigma)
    double getFeedMu();
    double getFeedSigma();

    // Computation
    double getFeed();
    double getFirstDerivativeFeed(const int &i1d);
    double getSecondDerivativeFeed(const int &i2d);
    double getVariationalFirstDerivativeFeed(const int &iv1d);
    double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d);
    double getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d);

    // randomizer implementations
    void randomizeBeta();
};

#endif
