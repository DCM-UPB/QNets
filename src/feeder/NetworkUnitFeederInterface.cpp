#include "NetworkUnitFeederInterface.hpp"
#include "NetworkUnit.hpp"
#include "FedNetworkUnit.hpp"

// --- StringCode methods

std::string NetworkUnitFeederInterface::getParams()
{
    return composeParamCode("id_shift", _vp_id_shift);
}


void NetworkUnitFeederInterface::setParams(const std::string &params)
{
    int starting_index;
    std::string str = readParamValue(params, "id_shift");
    if (setParamValue(str, starting_index)) this->setVariationalParametersIndexes(starting_index);
}

// set VP Indexes default version

int NetworkUnitFeederInterface::setVariationalParametersIndexes(const int &starting_index, const bool flag_add_vp){
    // Here we assign external vp indexes to internal indexes.
    // NOTE: The current method assumes, that no index larger than max_id,
    //       max_id = starting_index + source.size() - 1 ,
    //       may be in use FOR (and trivially IN) this feeder.

    _map_index_to_sources.clear();

    for (int j=0; j<starting_index; ++j) {
        std::vector<int> empty_vec;
        _map_index_to_sources.push_back(empty_vec);
    }

    for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i) {
        if(FedNetworkUnit * fu = dynamic_cast<FedNetworkUnit *>(_source[i])) {
            NetworkUnitFeederInterface * feeder = fu->getFeeder();
            if (feeder != 0) {
                for (int j=0; j<starting_index; ++j) {
                    if (feeder->isVPIndexUsedForFeeder(j)) {
                        _map_index_to_sources[j].push_back(i);
                    }
                }
            }
        }
    }

    _vp_id_shift = -1;
    return starting_index;
}




// --- is VP Index used

bool NetworkUnitFeederInterface::isVPIndexUsedInFeeder(const int &id){
    if ( _vp_id_shift <= id && id <_vp_id_shift+getNVariationalParameters()) {
        return true;
    }
    else return false;
}

bool NetworkUnitFeederInterface::isVPIndexUsedInSources(const int &id){
    if (id < _vp_id_shift) {
        return (!_map_index_to_sources[id].empty());
    }
    else return false;
}

bool NetworkUnitFeederInterface::isVPIndexUsedForFeeder(const int &id){
    if ( isVPIndexUsedInFeeder(id) || isVPIndexUsedInSources(id) ) {
        return true;
    }
    else return false;
}
