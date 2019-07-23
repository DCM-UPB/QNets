#ifndef FFNN_FMAP_ONEDIMSTATICMAP_HPP
#define FFNN_FMAP_ONEDIMSTATICMAP_HPP

#include "qnets/poly/feed/StaticFeeder.hpp"
#include "qnets/poly/layer/NetworkLayer.hpp"

#include <vector>

class OneDimStaticMap: public StaticFeeder
{
protected:
    size_t _nsrc; // number of source units, should be hardcoded by child in most cases

public:
    OneDimStaticMap(NetworkLayer * nl, const size_t &nsrc); // full initialization
    ~OneDimStaticMap() override = default;

    // string code methods
    std::string getParams() override;
    void setParams(const std::string &params) override;

    // parameter manipulation (child classes can use extra_params for extension)
    virtual void setParameters(const std::vector<size_t> &source_id0s, const std::vector<double> &extra_params = {});
};

#endif
