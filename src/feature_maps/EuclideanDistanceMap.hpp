#ifndef EUCLIDEAN_DISTANCE_MAP
#define EUCLIDEAN_DISTANCE_MAP

#include "StaticFeeder.hpp"
#include "NetworkLayer.hpp"

class EuclideanDistanceMap: public StaticFeeder
{
protected:
    int _ndim;

    double _calcDist(); // calculate euclidean distance

public:
    EuclideanDistanceMap(NetworkLayer * nl, const size_t &source_id1, const size_t &source_id2, const int ndim); // ids of the first index of relevant vectors
    EuclideanDistanceMap(NetworkLayer * nl): EuclideanDistanceMap(nl, 0, 0, 0) {} // minimal default initialization
    ~EuclideanDistanceMap(){}

    // string code methods
    std::string getIdCode(){return "EDM";} // return an identification string
    std::string getParams();
    void setParams(const std::string &params);

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
