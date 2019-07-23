#ifndef FFNN_FMAP_EUCLIDEANDISTANCEMAPUNIT_HPP
#define FFNN_FMAP_EUCLIDEANDISTANCEMAPUNIT_HPP

#include "qnets/poly/fmap/EuclideanDistanceMap.hpp"
#include "qnets/poly/fmap/FeatureMapUnit.hpp"

#include <string>

class EuclideanDistanceMapUnit: public FeatureMapUnit<EuclideanDistanceMap>
{
public:
    // string code id
    std::string getIdCode() override { return "EDMU"; } // return identifier for unit type
};

#endif
