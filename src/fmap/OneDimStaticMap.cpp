#include "ffnn/fmap/OneDimStaticMap.hpp"

// --- Constructor

OneDimStaticMap::OneDimStaticMap(NetworkLayer * nl, const size_t &nsrc): _nsrc(nsrc)
{
    _fillSourcePool(nl);
}

// --- StringCode methods

std::string OneDimStaticMap::getParams()
{
    std::string params = StaticFeeder::getParams();
    for (size_t i = 0; i < _nsrc; ++i) {
        params = composeCodes(params, composeParamCode("source_id" + std::to_string(i), _source_ids[i]));
    }
    return params;
}


void OneDimStaticMap::setParams(const std::string &params)
{
    std::string str;
    std::vector<size_t> source_ids;
    for (size_t i = 0; i < _nsrc; ++i) {
        size_t id;
        str = readParamValue(params, "source_id" + std::to_string(i));
        setParamValue(str, id);
        source_ids.push_back(id);
    }

    _fillSources(source_ids);
    StaticFeeder::setParams(params);
}


// --- Parameter manipulation

void OneDimStaticMap::setParameters(const std::vector<size_t> &source_id0s, const std::vector<double> & /*extra_params*/)
{
    std::vector<size_t> source_ids;

    for (size_t i = 0; i < _nsrc; ++i) {
        source_ids.push_back(source_id0s[i]);
    }

    _fillSources(source_ids);
    if (_vp_id_shift > -1) {
        this->setVariationalParametersIndexes(_vp_id_shift, false);
    }
}
