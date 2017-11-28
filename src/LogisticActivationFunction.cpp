#include "LogisticActivationFunction.hpp"

#include <math.h>



// Activation Function Interface implementation

double LogisticActivationFunction::f(const double &in)
{
   return (1./(1.+exp(-in)));
}


double LogisticActivationFunction::f1d(const double &in)
{
   double f=this->f(in);
   return (f*(1.-f));
}


double LogisticActivationFunction::f2d(const double &in)
{
   double f=this->f(in);
   return (f*(1.-f)*(1.-2.*f));
}
