#ifndef EUCLIDEAN_PAIR_DISTANCE_MAP_UNIT
#define EUCLIDEAN_PAIR_DISTANCE_MAP_UNIT

#include "FeatureMapUnit.hpp"
#include "EuclideanPairDistanceMap.hpp"

#include <string>

class EuclideanPairDistanceMapUnit: public FeatureMapUnit<EuclideanPairDistanceMap>
{
public:
    // string code id
    std::string getIdCode(){return "EPDMU";} // return identifier for unit type
};

#endif
