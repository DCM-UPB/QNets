#ifndef FFNN_FMAP_EUCLIDEANPAIRDISTANCEMAPUNIT_HPP
#define FFNN_FMAP_EUCLIDEANPAIRDISTANCEMAPUNIT_HPP

#include "ffnn/fmap/EuclideanPairDistanceMap.hpp"
#include "ffnn/fmap/FeatureMapUnit.hpp"

#include <string>

class EuclideanPairDistanceMapUnit: public FeatureMapUnit<EuclideanPairDistanceMap>
{
public:
    // string code id
    std::string getIdCode() override { return "EPDMU"; } // return identifier for unit type
};

#endif
