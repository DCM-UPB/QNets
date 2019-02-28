#ifndef FFNN_FMAP_PAIRDIFFERENCEMAPUNIT_HPP
#define FFNN_FMAP_PAIRDIFFERENCEMAPUNIT_HPP

#include "ffnn/fmap/FeatureMapUnit.hpp"
#include "ffnn/fmap/PairDifferenceMap.hpp"

#include <string>

class PairDifferenceMapUnit: public FeatureMapUnit<PairDifferenceMap>
{
public:
    // string code id
    std::string getIdCode() override{return "PDMU";} // return identifier for unit type
};

#endif
