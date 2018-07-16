#ifndef IDENTITY_MAP
#define IDENTITY_MAP

#include "FeederInterface.hpp"
#include "NetworkUnit.hpp"
#include "NetworkLayer.hpp"

#include <string>

class IdentityMap: public FeederInterface
{
public:
    IdentityMap(NetworkLayer * nl, const size_t &source_id);
    ~IdentityMap(){}

    // string code methods
    std::string getIdCode(){return "IDM";} // return an identification string

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
};

#endif
