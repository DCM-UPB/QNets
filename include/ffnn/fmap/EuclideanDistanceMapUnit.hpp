#ifndef FFNN_FMAP_EUCLIDEANDISTANCEMAPUNIT_HPP
#define FFNN_FMAP_EUCLIDEANDISTANCEMAPUNIT_HPP

#include "ffnn/fmap/EuclideanDistanceMap.hpp"
#include "ffnn/fmap/FeatureMapUnit.hpp"

#include <string>

class EuclideanDistanceMapUnit: public FeatureMapUnit<EuclideanDistanceMap>
{
public:
    // string code id
    std::string getIdCode() override { return "EDMU"; } // return identifier for unit type
};

#endif
