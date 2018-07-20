#ifndef EUCLIDEAN_DISTANCE_MAP_UNIT
#define EUCLIDEAN_DISTANCE_MAP_UNIT

#include "FedUnit.hpp"
#include "EuclideanDistanceMap.hpp"

#include <stdexcept>
#include <string>

class EuclideanDistanceMapUnit: public FedUnit
{
public:
    // Constructor and destructor
    EuclideanDistanceMapUnit(EuclideanDistanceMap * edmap = NULL) : FedUnit(static_cast<FeederInterface *>(edmap)) {}
    ~EuclideanDistanceMapUnit(){}

    // string code id
    std::string getIdCode(){return "EDMU";} // return identifier for unit type

    // restrict feeder to EuclideanDistanceMap
    void setFeeder(FeederInterface * feeder){
        if (EuclideanDistanceMap * edmap = dynamic_cast<EuclideanDistanceMap *>(feeder)) {
            FedUnit::setFeeder(edmap);
        }
        else {
            throw std::invalid_argument("[EuclideanDistanceMapUnit::setFeeder] Passed feeder is not of class EuclideanDistanceMap.");
        }
    }

    EuclideanDistanceMap * getEDMap(){return static_cast<EuclideanDistanceMap *>(_feeder);}

    // devirtualize
    void computeActivation(){}
    void computeValues(){NetworkUnit::computeValues();}
};

#endif
