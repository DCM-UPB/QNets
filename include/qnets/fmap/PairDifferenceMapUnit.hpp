#ifndef FFNN_FMAP_PAIRDIFFERENCEMAPUNIT_HPP
#define FFNN_FMAP_PAIRDIFFERENCEMAPUNIT_HPP

#include "qnets/fmap/FeatureMapUnit.hpp"
#include "qnets/fmap/PairDifferenceMap.hpp"

#include <string>

class PairDifferenceMapUnit: public FeatureMapUnit<PairDifferenceMap>
{
public:
    // string code id
    std::string getIdCode() override { return "PDMU"; } // return identifier for unit type
};

#endif
