#ifndef FFNN_FMAP_IDENTITYMAP_HPP
#define FFNN_FMAP_IDENTITYMAP_HPP

#include "qnets/poly/fmap/OneDimStaticMap.hpp"
#include "qnets/poly/layer/NetworkLayer.hpp"
#include "qnets/poly/unit/NetworkUnit.hpp"

#include <cstddef> // NULL
#include <string>

class IdentityMap: public OneDimStaticMap
{
protected:
    NetworkUnit * _src; // we gain some performance by maintaining and using this single pointer over vector with one element

    void _fillSources(const std::vector<size_t> &source_ids) override; // we extend this to maintain _src
    void _clearSources() override;

public:
    IdentityMap(NetworkLayer * nl, const size_t &source_id)
            : OneDimStaticMap(nl, 1), _src(nullptr) { setParameters(source_id); } // full initialization
    explicit IdentityMap(NetworkLayer * nl): IdentityMap(nl, 0) {} // minimal default initialization
    ~IdentityMap() override = default;

    // string code methods
    std::string getIdCode() override { return "IDM"; } // return an identification string

    // parameter manipulation
    void setParameters(const size_t &source_id); // calls base setParameters with vectorized argument

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
