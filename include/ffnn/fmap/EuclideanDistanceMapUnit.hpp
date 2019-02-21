#ifndef EUCLIDEAN_DISTANCE_MAP_UNIT
#define EUCLIDEAN_DISTANCE_MAP_UNIT

#include "ffnn/fmap/FeatureMapUnit.hpp"
#include "ffnn/fmap/EuclideanDistanceMap.hpp"

#include <string>

class EuclideanDistanceMapUnit: public FeatureMapUnit<EuclideanDistanceMap>
{
public:
    // string code id
    std::string getIdCode(){return "EDMU";} // return identifier for unit type
};

#endif
