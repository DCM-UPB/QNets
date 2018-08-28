#ifndef EUCLIDEAN_DISTANCE_MAP_UNIT
#define EUCLIDEAN_DISTANCE_MAP_UNIT

#include "FeatureMapUnit.hpp"
#include "EuclideanDistanceMap.hpp"

#include <string>

class EuclideanDistanceMapUnit: public FeatureMapUnit<EuclideanDistanceMap>
{
public:
    // string code id
    std::string getIdCode(){return "EDMU";} // return identifier for unit type
};

#endif
