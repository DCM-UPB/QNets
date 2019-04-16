#ifndef FFNN_FMAP_MULTIDIMSTATICMAP_HPP
#define FFNN_FMAP_MULTIDIMSTATICMAP_HPP

#include "ffnn/feed/StaticFeeder.hpp"
#include "ffnn/layer/NetworkLayer.hpp"

#include <vector>

class MultiDimStaticMap: public StaticFeeder
{
protected:
    size_t _ndim; // dimension of the used vectors
    size_t _nsrc; // number of source vectors, not individual units (should be hardcoded by child in most cases)

public:
    MultiDimStaticMap(NetworkLayer * nl, const size_t &ndim, const size_t &nsrc);
    ~MultiDimStaticMap() override = default;

    // string code methods
    std::string getParams() override;
    void setParams(const std::string &params) override;

    // parameter manipulation (child classes can use extra_params for extension)
    virtual void setParameters(const size_t &ndim, const std::vector<size_t> &source_id0s, const std::vector<double> &extra_params = {});
};

#endif
