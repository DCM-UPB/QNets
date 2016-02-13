#ifndef GAUSSIAN_ACTIVATION_FUNCTION
#define GAUSSIAN_ACTIVATION_FUNCTION

#include "ActivationFunctionInterface.hpp"

class GaussianActivationFunction: public ActivationFunctionInterface
{
protected:

public:

   double f(const double &in);

   double f1d(const double &in);

   double f2d(const double &in);
};


#endif
