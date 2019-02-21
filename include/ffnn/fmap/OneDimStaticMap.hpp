#ifndef ONE_DIM_STATIC_MAP
#define ONE_DIM_STATIC_MAP

#include "StaticFeeder.hpp"
#include "NetworkLayer.hpp"

#include <vector>

class OneDimStaticMap: public StaticFeeder
{
protected:
    size_t _nsrc; // number of source units, should be hardcoded by child in most cases

public:
    OneDimStaticMap(NetworkLayer * nl, const size_t &nsrc); // full initialization
    virtual ~OneDimStaticMap(){}

    // string code methods
    virtual std::string getParams();
    virtual void setParams(const std::string &params);

    // parameter manipulation (child classes can use extra_params for extension)
    virtual void setParameters(const std::vector<size_t> &source_id0s, const std::vector<double> &extra_params = {});
};

#endif
