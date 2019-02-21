#include "ffnn/feed/FeederInterface.hpp"
#include "ffnn/layer/NetworkLayer.hpp"
#include "ffnn/unit/NetworkUnit.hpp"
#include "ffnn/unit/FedUnit.hpp"
#include <vector>

// --- Base Destructor

void FeederInterface::_clearSources()
{
    _sources.clear();
    _source_ids.clear();
    _map_index_to_sources.clear();
}

FeederInterface::~FeederInterface()
{
    _sourcePool.clear();
    FeederInterface::_clearSources();
}


// --- fillSources methods

void FeederInterface::_fillSourcePool(NetworkLayer * nl)
{
    _sourcePool.clear();
    _clearSources();
    for (int i=0; i<nl->getNUnits(); ++i){
        _sourcePool.push_back(nl->getUnit(i));
    }
}

void FeederInterface::_fillSources(const std::vector<size_t> &source_ids) // add select sources from sourcePool
{
    _clearSources();
    for (size_t i=0; i<source_ids.size(); ++i) {
        _source_ids.push_back(source_ids[i]);
        _sources.push_back(_sourcePool[source_ids[i]]);
    }
}

void FeederInterface::_fillSources() // add all sources from sourcePool
{
    _clearSources();
    for (size_t i=0; i<_sourcePool.size(); ++i) {
        _source_ids.push_back(i);
        _sources.push_back(_sourcePool[i]);
    }
}


// --- StringCode methods

std::string FeederInterface::getParams()
{
    return composeParamCode("id_shift", _vp_id_shift);
}

void FeederInterface::setParams(const std::string &params)
{
    std::string str_id = readParamValue(params, "id_shift");
    setParamValue(str_id, _vp_id_shift);
    // in the child class you need to extend this and call setVariationalParametersIndexes after having all information
}


// set VP Indexes default version

int FeederInterface::setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp){
    // Here we assign external vp indexes to internal indexes.
    // NOTE: The current method assumes, that no index relevant
    // to this feeder is larger than starting_index (+ n_own_vp - 1)
    // NOTE 2: Extend this to actually add variational parameters

    _map_index_to_sources.clear();

    for (int j=0; j<starting_index; ++j) {
        std::vector<size_t> empty_vec;
        _map_index_to_sources.push_back(empty_vec);
    }

    for (std::vector<NetworkUnit *>::size_type i=1; i<_sources.size(); ++i) {
        if(FedUnit * fu = dynamic_cast<FedUnit *>(_sources[i])) {
            FeederInterface * feeder = fu->getFeeder();
            if (feeder != 0) {
                for (int j=0; j<starting_index; ++j) {
                    if (feeder->isVPIndexUsedForFeeder(j)) {
                        _map_index_to_sources[j].push_back(i);
                    }
                }
            }
        }
    }

    _vp_id_shift = starting_index;
    return starting_index;
}


bool FeederInterface::isVPIndexUsedInSources(const int &id)
{
    if (id < _vp_id_shift) {
        return (!_map_index_to_sources[id].empty());
    }
    else return false;
}

bool FeederInterface::isVPIndexUsedForFeeder(const int &id)
{
    if ( isVPIndexUsedInFeeder(id) || isVPIndexUsedInSources(id) ) {
        return true;
    }
    else return false;
}
