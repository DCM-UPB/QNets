#ifndef EUCLIDEAN_DISTANCE_MAP
#define EUCLIDEAN_DISTANCE_MAP

#include "MultiDimStaticMap.hpp"
#include "NetworkLayer.hpp"

#include <vector>
#include <cstddef>

class EuclideanDistanceMap: public MultiDimStaticMap
{
protected:
    std::vector<double> _fixedPoint; // ndim-dimensional coordinate to calculate distance against

    double _calcDist(); // calculate euclidean distance

public:
    EuclideanDistanceMap(NetworkLayer * nl, const size_t ndim, const size_t &source_id0, const std::vector<double> fixedPoint)
        : MultiDimStaticMap(nl, ndim, 1) {setParameters(ndim, source_id0, fixedPoint);} // full initialization;
    EuclideanDistanceMap(NetworkLayer * nl): EuclideanDistanceMap(nl, 1, 0, std::vector<double> {0.} ) {} // minimal default initialization
    ~EuclideanDistanceMap(){}

    // string code methods
    std::string getIdCode(){return "EDM";} // return an identification string

    // parameter manipulation
    void setParameters(const size_t &ndim, const size_t &source_id0, const std::vector<double> &fixedPoint);
    void setParameters(const size_t &ndim, const vector<size_t> &source_id0s, const std::vector<double> &fixedPoint); // overriding

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
