#ifndef IDENTITY_MAP
#define IDENTITY_MAP

#include "OneDimStaticMap.hpp"
#include "NetworkUnit.hpp"
#include "NetworkLayer.hpp"

#include <string>
#include <cstddef> // NULL

class IdentityMap: public OneDimStaticMap
{
protected:
    NetworkUnit * _src; // we gain some performance by maintaining and using this single pointer over vector with one element

    void _fillSources(const std::vector<size_t> &source_ids); // we extend this to maintain _src
    void _clearSources();

public:
    IdentityMap(NetworkLayer * nl, const size_t &source_id)
        : OneDimStaticMap(nl, 1), _src(NULL) {setParameters(source_id);} // full initialization
    explicit IdentityMap(NetworkLayer * nl): IdentityMap(nl, 0) {} // minimal default initialization
    ~IdentityMap(){}

    // string code methods
    std::string getIdCode(){return "IDM";} // return an identification string

    // parameter manipulation
    void setParameters(const size_t &source_id); // calls base setParameters with vectorized argument

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
