#ifndef FFNN_FMAP_PAIRSUMMAPUNIT_HPP
#define FFNN_FMAP_PAIRSUMMAPUNIT_HPP

#include "ffnn/fmap/FeatureMapUnit.hpp"
#include "ffnn/fmap/PairSumMap.hpp"

#include <string>

class PairSumMapUnit: public FeatureMapUnit<PairSumMap>
{
public:
    // string code id
    std::string getIdCode() override{return "PSMU";} // return identifier for unit type
};

#endif
