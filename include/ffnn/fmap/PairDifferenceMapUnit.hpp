#ifndef PAIR_DIFFERENCE_MAP_UNIT
#define PAIR_DIFFERENCE_MAP_UNIT

#include "ffnn/fmap/FeatureMapUnit.hpp"
#include "ffnn/fmap/PairDifferenceMap.hpp"

#include <string>

class PairDifferenceMapUnit: public FeatureMapUnit<PairDifferenceMap>
{
public:
    // string code id
    std::string getIdCode(){return "PDMU";} // return identifier for unit type
};

#endif
