#ifndef FFNN_FMAP_EUCLIDEANDISTANCEMAP_HPP
#define FFNN_FMAP_EUCLIDEANDISTANCEMAP_HPP

#include "qnets/poly/fmap/MultiDimStaticMap.hpp"
#include "qnets/poly/layer/NetworkLayer.hpp"

#include <cstddef>
#include <vector>

// takes coordinates on input side and calculates squared euclidean distance against a fixed point
class EuclideanDistanceMap: public MultiDimStaticMap
{
protected:
    std::vector<double> _fixedPoint; // ndim-dimensional coordinate to calculate distance against

    double _calcDist(); // calculate euclidean distance

public:
    EuclideanDistanceMap(NetworkLayer * nl, const size_t ndim, const size_t &source_id0, const std::vector<double> &fixedPoint)
            : MultiDimStaticMap(nl, ndim, 1) { setParameters(ndim, source_id0, fixedPoint); } // full initialization;
    explicit EuclideanDistanceMap(NetworkLayer * nl):
            EuclideanDistanceMap(nl, 1, 0, std::vector<double>{0.}) {} // minimal default initialization
    ~EuclideanDistanceMap() override = default;

    // string code methods
    std::string getIdCode() override { return "EDM"; } // return an identification string

    // string code methods
    std::string getParams() override;
    void setParams(const std::string &params) override;

    // parameter manipulation
    void setParameters(const size_t &ndim, const size_t &source_id0, const std::vector<double> &fixedPoint);
    void setParameters(const size_t &ndim, const vector<size_t> &source_id0s, const std::vector<double> &fixedPoint) override; // overriding

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
