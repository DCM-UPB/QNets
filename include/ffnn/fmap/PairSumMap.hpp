#ifndef FFNN_FMAP_PAIRSUMMAP_HPP
#define FFNN_FMAP_PAIRSUMMAP_HPP

#include "ffnn/fmap/OneDimStaticMap.hpp"
#include "ffnn/layer/NetworkLayer.hpp"
#include "ffnn/unit/NetworkUnit.hpp"

#include <cstddef> // NULL
#include <string>

class PairSumMap: public OneDimStaticMap
{
protected:
    // instead of the two element vector we use these pointers internally, for better performance
    NetworkUnit * _src0;
    NetworkUnit * _src1;

    void _fillSources(const std::vector<size_t> &source_ids) override; // we extend this to maintain _src1/2
    void _clearSources() override;

public:
    PairSumMap(NetworkLayer * nl, const size_t &source_id0, const size_t &source_id1)
            :
            OneDimStaticMap(nl, 2), _src0(nullptr), _src1(nullptr) { setParameters(source_id0, source_id1); } // full initialization
    explicit PairSumMap(NetworkLayer * nl): PairSumMap(nl, 0, 0) {} // minimal default initialization
    ~PairSumMap() override = default;

    // string code methods
    std::string getIdCode() override { return "PSM"; } // return an identification string

    // parameter manipulation
    void setParameters(const size_t &source_id0, const size_t &source_id1); // calls base setParameters with vectorized argument

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
