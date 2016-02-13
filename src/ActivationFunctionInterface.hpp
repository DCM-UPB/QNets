#ifndef ACTIVATION_FUNCTION_INTERFACE
#define ACTIVATION_FUNCTION_INTERFACE


class ActivationFunctionInterface
{
protected:

public:

   // compute the activation function value
   virtual double f(const double &) = 0;

   // first derivative of the activation function
   virtual double f1d(const double &) = 0;

   // second derivative of the activation function
   virtual double f2d(const double &) = 0;
};


#endif
