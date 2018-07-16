#ifndef IDENTITY_MAP_UNIT
#define IDENTITY_MAP_UNIT

#include "FedUnit.hpp"
#include "IdentityMap.hpp"

#include <stdexcept>
#include <string>

class IdentityMapUnit: public FedUnit
{
public:
    // Constructor and destructor
    IdentityMapUnit(IdentityMap * idmap = NULL) : FedUnit(static_cast<FeederInterface *>(idmap)) {}
    virtual ~IdentityMapUnit(){}

    // string code id
    virtual std::string getIdCode(){return "IDMU";} // return identifier for unit type

    // restrict feeder to IdentityMap
    void setFeeder(FeederInterface * feeder){
        if (IdentityMap * idmap = dynamic_cast<IdentityMap *>(feeder)) {
            FedUnit::setFeeder(idmap);
        }
        else {
            throw std::invalid_argument("[IdentityMapUnit::setFeeder] Passed feeder is not of class IdentityMap.");
        }
    }
};


#endif
