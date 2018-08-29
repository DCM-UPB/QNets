#ifndef EUCLIDEAN_PAIR_DISTANCE_MAP
#define EUCLIDEAN_PAIR_DISTANCE_MAP

#include "MultiDimStaticMap.hpp"
#include "NetworkLayer.hpp"

// takes coordinates on input side and calculates squared euclidean distance of a pair
class EuclideanPairDistanceMap: public MultiDimStaticMap
{
protected:
    double _calcDist(); // calculate euclidean distance

public:
    EuclideanPairDistanceMap(NetworkLayer * nl, const size_t ndim, const size_t &source_id0, const size_t &source_id1)
        : MultiDimStaticMap(nl, ndim, 2) {setParameters(ndim, source_id0, source_id1);} // full initialization;
    explicit EuclideanPairDistanceMap(NetworkLayer * nl): EuclideanPairDistanceMap(nl, 0, 0, 0) {} // minimal default initialization
    ~EuclideanPairDistanceMap(){}

    // string code methods
    std::string getIdCode(){return "EPDM";} // return an identification string

    // parameter manipulation
    void setParameters(const size_t &ndim, const size_t &source_id0, const size_t &source_id1);

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
