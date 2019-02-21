#ifndef MULTI_DIM_STATIC_MAP
#define MULTI_DIM_STATIC_MAP

#include "StaticFeeder.hpp"
#include "NetworkLayer.hpp"

#include <vector>

class MultiDimStaticMap: public StaticFeeder
{
protected:
    size_t _ndim; // dimension of the used vectors
    size_t _nsrc; // number of source vectors, not individual units (should be hardcoded by child in most cases)

public:
    MultiDimStaticMap(NetworkLayer * nl, const size_t &ndim, const size_t &nsrc);
    virtual ~MultiDimStaticMap(){}

    // string code methods
    virtual std::string getParams();
    virtual void setParams(const std::string &params);

    // parameter manipulation (child classes can use extra_params for extension)
    virtual void setParameters(const size_t &ndim, const std::vector<size_t> &source_id0s, const std::vector<double> &extra_params = {});
};

#endif
