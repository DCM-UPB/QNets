#ifndef IDENTITY_ACTIVATION_FUNCTION
#define IDENTITY_ACTIVATION_FUNCTION


class IdentityActivationFunction: public ActivationFunctionInterface
{
protected:

public:
   double f(const double &in){return in;}
   double f1d(const double &in){return 1.;}
   double f2d(const double &in){return 0.;}
};


#endif
