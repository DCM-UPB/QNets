#ifndef EUCLIDEAN_DISTANCE_MAP
#define EUCLIDEAN_DISTANCE_MAP

#include "MultiDimStaticMap.hpp"
#include "NetworkLayer.hpp"

class EuclideanDistanceMap: public MultiDimStaticMap
{
protected:
    double _calcDist(); // calculate euclidean distance

public:
    EuclideanDistanceMap(NetworkLayer * nl, const size_t ndim, const size_t &source_id1, const size_t &source_id2)
        : MultiDimStaticMap(nl, ndim, 2) {setParameters(ndim, source_id1, source_id2);} // full initialization;
    EuclideanDistanceMap(NetworkLayer * nl): EuclideanDistanceMap(nl, 0, 0, 0) {} // minimal default initialization
    ~EuclideanDistanceMap(){}

    // string code methods
    std::string getIdCode(){return "EDM";} // return an identification string

    // parameter manipulation
    void setParameters(const size_t &ndim, const size_t &source_id1, const size_t &source_id2);

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
