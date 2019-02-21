#ifndef PAIR_SUM_MAP_UNIT
#define PAIR_SUM_MAP_UNIT

#include "FeatureMapUnit.hpp"
#include "PairSumMap.hpp"

#include <string>

class PairSumMapUnit: public FeatureMapUnit<PairSumMap>
{
public:
    // string code id
    std::string getIdCode(){return "PSMU";} // return identifier for unit type
};

#endif
