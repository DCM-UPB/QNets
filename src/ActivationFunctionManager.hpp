#ifndef ACTIVATION_FUNCTION_MANAGER
#define ACTIVATION_FUNCTION_MANAGER


#include "ActivationFunctionInterface.hpp"

#include <string>


class ActivationFunctionManager{
public:
   static ActivationFunctionInterface * provideActivationFunction(const std::string idcode);
};



#endif