#ifndef LOGISTIC_ACTIVATION_FUNCTION
#define LOGISTIC_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"

class LogisticActivationFunction: public ActivationFunctionInterface
{
protected:

public:

   double f(const double &in);

   double f1d(const double &in);

   double f2d(const double &in);
};


#endif
