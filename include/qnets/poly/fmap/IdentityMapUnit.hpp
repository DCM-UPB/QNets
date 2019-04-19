#ifndef FFNN_FMAP_IDENTITYMAPUNIT_HPP
#define FFNN_FMAP_IDENTITYMAPUNIT_HPP

#include "qnets/poly/fmap/FeatureMapUnit.hpp"
#include "qnets/poly/fmap/IdentityMap.hpp"

#include <string>

class IdentityMapUnit: public FeatureMapUnit<IdentityMap>
{
public:
    // string code id
    std::string getIdCode() override { return "IDMU"; } // return identifier for unit type
};

#endif
