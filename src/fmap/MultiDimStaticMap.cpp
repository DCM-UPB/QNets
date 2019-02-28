#include "ffnn/fmap/MultiDimStaticMap.hpp"
#include "ffnn/serial/StringCodeUtilities.hpp"
#include "ffnn/unit/NetworkUnit.hpp"

#include <string>
#include <vector>

// --- Constructor

MultiDimStaticMap::MultiDimStaticMap(NetworkLayer * nl, const size_t &ndim, const size_t &nsrc): _ndim(ndim), _nsrc(nsrc)
{
    _fillSourcePool(nl);
}

// --- StringCode methods

std::string MultiDimStaticMap::getParams()
{
    std::string params = composeParamCode("ndim", _ndim);
    for (size_t i=0; i<_nsrc; ++i) {
        params = composeCodes(params, composeParamCode("source_id" + std::to_string(i), _source_ids[i*_ndim]));
    }
    return composeCodes(StaticFeeder::getParams(), params);
}


void MultiDimStaticMap::setParams(const std::string &params)
{
    std::string str;
    str = readParamValue(params, "ndim");
    setParamValue(str, _ndim);

    std::vector<size_t> source_ids;
    for (size_t i=0; i<_nsrc; ++i) {
        size_t id;
        str = readParamValue(params, "source_id" + std::to_string(i));
        setParamValue(str, id);
        for (size_t j=0; j<_ndim; ++j) { source_ids.push_back(id+j);
}
    }

    _fillSources(source_ids);
    StaticFeeder::setParams(params);
}


// --- Parameter manipulation

void MultiDimStaticMap::setParameters(const size_t &ndim, const std::vector<size_t> &source_id0s, const std::vector<double> & /*extra_params*/)
{
    std::vector<size_t> source_ids;

    _ndim = ndim;

    for (size_t i=0; i<_nsrc; ++i) {
        // just fill with offset indices in the ndim=0 case (i.e. disabled/uninitialized)
        source_ids.push_back(_ndim>0 ? source_id0s[i] : 0);
        for (size_t j=1; j<_ndim; ++j) {
            source_ids.push_back(source_id0s[i]+j);
        }
    }

    _fillSources(source_ids);
    if (_vp_id_shift > -1) { this->setVariationalParametersIndexes(_vp_id_shift, false);
}
}
