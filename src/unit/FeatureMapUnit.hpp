#ifndef FEATURE_MAP_UNIT
#define FEATURE_MAP_UNIT

#include "NetworkUnit.hpp"
#include "FeederInterface.hpp"
#include "FeatureMapFeeder.hpp"

#include <string>
#include <cstddef> // for NULL

// Network Unit with Feeder
class FeatureMapUnit: public FedNetworkUnit
{
public:
    // Constructor and destructor
    explicit FeatureMapUnit(FeatureMapFeeder * map = NULL): FedNetworkUnit(static_cast<FeederInterface *>(map)) {}
    virtual ~FedNetworkUnit(){}

    // setFeeder needs to be made safe
};

#endif
