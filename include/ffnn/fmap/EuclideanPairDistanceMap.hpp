#ifndef FFNN_FMAP_EUCLIDEANPAIRDISTANCEMAP_HPP
#define FFNN_FMAP_EUCLIDEANPAIRDISTANCEMAP_HPP

#include "ffnn/fmap/MultiDimStaticMap.hpp"
#include "ffnn/layer/NetworkLayer.hpp"

// takes coordinates on input side and calculates squared euclidean distance of a pair
class EuclideanPairDistanceMap: public MultiDimStaticMap
{
protected:
    double _calcDist(); // calculate euclidean distance

public:
    EuclideanPairDistanceMap(NetworkLayer * nl, const size_t ndim, const size_t &source_id0, const size_t &source_id1)
        : MultiDimStaticMap(nl, ndim, 2) {setParameters(ndim, source_id0, source_id1);} // full initialization;
    explicit EuclideanPairDistanceMap(NetworkLayer * nl): EuclideanPairDistanceMap(nl, 0, 0, 0) {} // minimal default initialization
    ~EuclideanPairDistanceMap() override= default;

    // string code methods
    std::string getIdCode() override{return "EPDM";} // return an identification string

    // parameter manipulation
    void setParameters(const size_t &ndim, const size_t &source_id0, const size_t &source_id1);

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
};

#endif
